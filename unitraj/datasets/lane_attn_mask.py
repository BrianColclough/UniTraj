import zlib

import numpy as np


def reach(links, hops):
    """
    Args:
        links (n, n): bool, symmetric, False on the diagonal
        hops: how many links a path may cross
    Returns:
        (n, n) bool, True where node j is within `hops` links of node i; every node reaches itself
    """
    step = (links | np.eye(len(links), dtype=bool)).astype(np.int32)
    within = np.eye(len(links), dtype=np.int32)
    for _ in range(hops):
        within = ((within @ step) > 0).astype(np.int32)
    return within.astype(bool)


def rewire(links, seed, swaps_per_link, tries_per_swap=50):
    """
    Degree-preserving double-edge swaps (Maslov & Sneppen 2002): links a-b and c-d become a-d and c-b, and a swap
    that would give a self-link or a duplicate link is rejected, so every node keeps its number of links. The same
    algorithm as rewire() in the setup repository's decision4_random_control.ipynb, which Decision 4's numbers come
    from.
    Args:
        links (n, n): bool, symmetric, False on the diagonal
        seed: seed of the swaps' own random generator, so the global numpy state is not touched
        swaps_per_link: successful swaps wanted, per link
        tries_per_swap: attempts allowed per wanted swap before giving up
    Returns:
        the rewired (n, n) graph, and a dict with the number of links, the swap target, the swaps and attempts made,
        and status: 'ok' (target reached), 'partly rewired' or 'unchanged' (the attempts ran out first, which
        includes graphs where no swap is legal), or 'not rewirable' (fewer than two links, or complete or one link
        short of it, which are skipped without trying)
    """
    n = len(links)
    iu, ju = np.nonzero(np.triu(links, 1))
    num_links = len(iu)
    target = swaps_per_link * num_links
    info = {'links': num_links, 'target': target, 'swaps': 0, 'tries': 0}
    if num_links < 2 or num_links >= n * (n - 1) // 2 - 1:
        info['status'] = 'not rewirable'
        return links.copy(), info
    adj = [set() for _ in range(n)]
    for a, b in zip(iu.tolist(), ju.tolist()):
        adj[a].add(b)
        adj[b].add(a)
    edges = [[a, b] for a, b in zip(iu.tolist(), ju.tolist())]
    rng = np.random.default_rng(seed)
    max_tries = tries_per_swap * target
    swaps = tries = 0
    while swaps < target and tries < max_tries:
        m = min(4096, max_tries - tries)
        e1s = rng.integers(num_links, size=m).tolist()
        e2s = rng.integers(num_links, size=m).tolist()
        flips = rng.integers(2, size=m).tolist()
        for e1, e2, flip in zip(e1s, e2s, flips):
            tries += 1
            if e1 != e2:
                a, b = edges[e1]
                c, d = edges[e2]
                if flip:
                    c, d = d, c
                if a != d and c != b and d not in adj[a] and b not in adj[c]:
                    adj[a].remove(b); adj[b].remove(a); adj[c].remove(d); adj[d].remove(c)
                    adj[a].add(d); adj[d].add(a); adj[c].add(b); adj[b].add(c)
                    edges[e1] = [a, d]
                    edges[e2] = [c, b]
                    swaps += 1
                    if swaps == target:
                        break
    info.update(swaps=swaps, tries=tries,
                status='ok' if swaps == target else ('partly rewired' if swaps else 'unchanged'))
    rewired = np.zeros((n, n), dtype=bool)
    ea = np.array(edges)
    rewired[ea[:, 0], ea[:, 1]] = True
    rewired[ea[:, 1], ea[:, 0]] = True
    return rewired, info


def mask_seed(scenario_id, track_id, seed):
    # a stable hash, unlike Python's hash(), so a sample gets the same random graph in every epoch, worker and process
    return zlib.crc32(f'{scenario_id}|{track_id}|{seed}'.encode())


def build_lane_attn_mask(lane_id, adjacency, condition, hops=None, relation_bits=None, swaps_per_link=None,
                         seed=None):
    """
    Which map tokens of one sample may attend to which in the lane-to-lane layer. The lanes that give a token are the
    graph's nodes: tokens of one lane are one node, so they are always open to each other and never count as
    partners, and a hop never passes through a lane with no token (Decision 3).
    Args:
        lane_id (S,): int64 lane ID of each map token, -1 where the slot holds no lane (map_polylines_lane_id)
        adjacency (S, S): uint8 LANE_* bits between tokens (map_polylines_adjacency)
        condition: 'learned', every lane pair; 'structured', lanes within `hops` links of each other, where a link is
            any relation in `relation_bits`, in either direction (Decisions 1 and 3); 'random', the structured graph
            rewired with `swaps_per_link` double-edge swaps per link, seeded by `seed` (Decision 4)
    Returns:
        (S, S) bool, True where token i may attend to token j and False in the row and column of every empty slot,
        and for 'random' the info dict of rewire() (None otherwise)
    """
    tokens = np.flatnonzero(lane_id >= 0)
    _, first, node = np.unique(lane_id[tokens], return_index=True, return_inverse=True)
    info = None
    if condition not in ('learned', 'structured', 'random'):
        raise ValueError(f'No lane-to-lane mask for condition {condition}')
    if condition == 'learned':
        lane_mask = np.ones((len(first), len(first)), dtype=bool)
    else:
        # get_lane_adjacency() gives every token of a lane that lane's links, so the lane's first token stands for it
        links = (adjacency[np.ix_(tokens[first], tokens[first])] & relation_bits) != 0
        links |= links.T
        np.fill_diagonal(links, False)
        lane_mask = reach(links, hops)
        if condition == 'random':
            # self entries are not partners, so they are taken out before rewiring and put back after it
            np.fill_diagonal(lane_mask, False)
            lane_mask, info = rewire(lane_mask, seed, swaps_per_link)
            np.fill_diagonal(lane_mask, True)
    mask = np.zeros((len(lane_id), len(lane_id)), dtype=bool)
    mask[np.ix_(tokens, tokens)] = lane_mask[np.ix_(node, node)]
    return mask, info
