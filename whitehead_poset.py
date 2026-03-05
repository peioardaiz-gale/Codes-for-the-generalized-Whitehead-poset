"""
Gamma-Whitehead Poset: Vertex Types, Essential Counts, Simplices, and E^1 page
===============================================================================
Usage:
    python3 whitehead_poset.py

Vertices are integers. Edges are lists of pairs (u, v).
Edit the EXAMPLES section at the bottom to define your own graph.

The geometric realisation associates an n-simplex to each strictly
increasing chain  tau_0 < tau_1 < ... < tau_n  of length n+1 in the
poset. The rank of the simplex is defined as rk(tau_0).

The stabiliser of each p-simplex sigma = (tau_0 < ... < tau_p) is free
abelian of rank r = rk(tau_0), so:

    H_q(Stab(sigma); Z) = Z^C(r, q)    where C(r,q) = binomial(r, q)

The E^1 page of the equivariant spectral sequence is:

    E^1_{p,q} = Z^{ sum_{sigma p-simplex} C(rk(tau_0), q) }

and we record the rank of each entry (as a free abelian group).
"""

from itertools import product as iproduct
from collections import defaultdict
from math import comb


# ── Graph utilities ────────────────────────────────────────────────────────────

def star(v, edges, vertices):
    """Return st(v) = {v} union lk(v)."""
    lk = {b for a, b in edges if a == v}
    lk |= {a for a, b in edges if b == v}
    return frozenset(lk | {v})


def get_components(verts, edges):
    """Connected components of the subgraph induced by verts."""
    remaining = set(verts)
    adj = defaultdict(set)
    for a, b in edges:
        if a in remaining and b in remaining:
            adj[a].add(b)
            adj[b].add(a)
    visited = set()
    components = []
    for v in sorted(remaining):
        if v in visited:
            continue
        comp = set()
        stack = [v]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            comp.add(node)
            stack.extend(adj[node] - visited)
        components.append(frozenset(comp))
    return components


def get_based_partitions(v, vertices, edges):
    """
    All valid based partitions with operative factor v.
    Each partition is a tuple of frozensets (petals), where each petal is a
    union of connected components of Gamma - st(v).
    """
    st_v = star(v, edges, vertices)
    remaining = frozenset(vertices) - st_v
    comps = get_components(remaining, edges)

    if not comps:
        return [()]  # trivial: no petals

    def all_set_partitions(lst):
        if not lst:
            yield []
            return
        first, *rest = lst
        for partition in all_set_partitions(rest):
            for i in range(len(partition)):
                new_p = [set(g) for g in partition]
                new_p[i].add(first)
                yield [frozenset(g) for g in new_p]
            yield partition + [frozenset({first})]

    result = []
    for raw in all_set_partitions(comps):
        petal_tuple = tuple(sorted(
            [frozenset().union(*group) for group in raw],
            key=lambda s: sorted(s)
        ))
        result.append(petal_tuple)
    return result


def minimal_component(v, vertices, edges):
    """The component of Gamma - st(v) containing the smallest vertex."""
    st_v = star(v, edges, vertices)
    remaining = frozenset(vertices) - st_v
    comps = get_components(remaining, edges)
    if not comps:
        return None
    return min(comps, key=lambda c: min(c))


# ── Crossing and compatibility ─────────────────────────────────────────────────

def is_adjacent(u, v, edges):
    return (u, v) in edges or (v, u) in edges


def shared_components(u, v, vertices, edges):
    st_u = star(u, edges, vertices)
    st_v = star(v, edges, vertices)
    comps_u = set(get_components(frozenset(vertices) - st_u, edges))
    comps_v = set(get_components(frozenset(vertices) - st_v, edges))
    return comps_u & comps_v


def dominant_component(u, v, vertices, edges):
    st_u = star(u, edges, vertices)
    remaining = frozenset(vertices) - st_u
    for c in get_components(remaining, edges):
        if v in c:
            return c
    return None


def do_cross(u, tau_u, v, tau_v, vertices, edges):
    if is_adjacent(u, v, edges) or u == v:
        return False
    shared = shared_components(u, v, vertices, edges)
    if not shared:
        return False
    D_v = dominant_component(u, v, vertices, edges)
    D_u = dominant_component(v, u, vertices, edges)
    for P in tau_u:
        for Q in tau_v:
            if D_v is not None and D_v.issubset(P):
                continue
            if D_u is not None and D_u.issubset(Q):
                continue
            for C in shared:
                if C.issubset(P) and C.issubset(Q):
                    return True
    return False


def are_compatible(u, tau_u, v, tau_v, vertices, edges):
    if u == v or is_adjacent(u, v, edges):
        return True
    return not do_cross(u, tau_u, v, tau_v, vertices, edges)


# ── Order relation ─────────────────────────────────────────────────────────────

def vertex_type_leq(combo_a, combo_b):
    """combo_a <= combo_b: every petal of combo_b[i] sits inside a petal of combo_a[i]."""
    for pa, pb in zip(combo_a, combo_b):
        for petal_b in pb:
            if not any(petal_b.issubset(petal_a) for petal_a in pa):
                return False
    return True


def vertex_type_lt(combo_a, combo_b):
    return combo_a != combo_b and vertex_type_leq(combo_a, combo_b)


def vertex_type_rank(combo):
    return sum(len(p) - 1 for p in combo)


# ── Essentiality ───────────────────────────────────────────────────────────────

def can_split_petal_compatibly(petal_idx, partition, v, combo,
                                vertices_list, vertices, edges):
    st_v = star(v, edges, vertices)
    remaining = frozenset(vertices) - st_v
    comps = get_components(remaining, edges)
    comps_in_petal = [c for c in comps if c.issubset(partition[petal_idx])]

    if len(comps_in_petal) < 2:
        return False

    n = len(comps_in_petal)
    v_idx = vertices_list.index(v)

    for mask in range(1, 1 << (n - 1)):
        group1 = frozenset().union(*[comps_in_petal[i] for i in range(n) if     mask & (1 << i)])
        group2 = frozenset().union(*[comps_in_petal[i] for i in range(n) if not mask & (1 << i)])

        new_partition = tuple(sorted(
            partition[:petal_idx] + (group1, group2) + partition[petal_idx + 1:],
            key=lambda s: sorted(s)
        ))

        compatible = all(
            are_compatible(v, new_partition, vertices_list[j], combo[j], vertices, edges)
            for j in range(len(vertices_list)) if j != v_idx
        )
        if compatible:
            return True

    return False


def is_essential_vertex_type(combo, vertices_list, vertices, edges):
    for i, v in enumerate(vertices_list):
        min_comp = minimal_component(v, vertices, edges)
        partition = combo[i]
        for petal_idx, petal in enumerate(partition):
            if can_split_petal_compatibly(petal_idx, partition, v, combo,
                                          vertices_list, vertices, edges):
                if min_comp is None or not min_comp.issubset(petal):
                    return False
    return True


# ── Main poset computation ─────────────────────────────────────────────────────

def compute_whitehead_poset(vertices, edges):
    """
    Compute the Gamma-Whitehead poset.
    Returns:
        valid_types    : list of all valid vertex types
        rank_counts    : dict  rank -> number of vertex types
        rank_essential : dict  rank -> number of essential vertex types
    """
    vertices = sorted(vertices)
    edges = [(min(a, b), max(a, b)) for a, b in edges]

    all_partitions = [get_based_partitions(v, vertices, edges) for v in vertices]

    valid_types    = []
    rank_counts    = defaultdict(int)
    rank_essential = defaultdict(int)

    for combo in iproduct(*all_partitions):
        compatible = True
        for i in range(len(vertices)):
            if not compatible:
                break
            for j in range(i + 1, len(vertices)):
                if not are_compatible(vertices[i], combo[i],
                                      vertices[j], combo[j],
                                      vertices, edges):
                    compatible = False
                    break
        if compatible:
            valid_types.append(combo)
            rank = vertex_type_rank(combo)
            rank_counts[rank] += 1
            if is_essential_vertex_type(combo, vertices, vertices, edges):
                rank_essential[rank] += 1

    return valid_types, dict(rank_counts), dict(rank_essential)


# ── Simplex computation ────────────────────────────────────────────────────────

def compute_simplices(valid_types):
    """
    Enumerate all simplices in the geometric realisation.
    An n-simplex <-> strictly increasing chain tau_0 < ... < tau_n.
    Its rank is rk(tau_0).

    Returns:
        simplex_counts : dict (n, bottom_rank) -> number of n-simplices
    """
    by_rank = defaultdict(list)
    for i, vt in enumerate(valid_types):
        by_rank[vertex_type_rank(vt)].append((i, vt))
    ranks = sorted(by_rank.keys())

    idx_of = {vt: i for i, vt in enumerate(valid_types)}
    simplex_counts = defaultdict(int)

    def dfs(chain, top_idx, top_rank):
        n           = len(chain) - 1
        bottom_rank = vertex_type_rank(chain[0])
        simplex_counts[(n, bottom_rank)] += 1
        for r in ranks:
            if r <= top_rank:
                continue
            for j, vt in by_rank[r]:
                if j > top_idx and vertex_type_lt(chain[-1], vt):
                    dfs(chain + [vt], j, r)

    for i, vt in enumerate(valid_types):
        dfs([vt], i, vertex_type_rank(vt))

    return dict(simplex_counts)


# ── E^1 page computation ───────────────────────────────────────────────────────

def compute_E1_page(simplex_counts):
    """
    Compute the E^1 page of the equivariant spectral sequence.

    The stabiliser of a p-simplex with bottom rank r is Z^r (free abelian),
    whose homology is:
        H_q(Z^r; Z) = Z^C(r, q)    [C(r,q) = binomial coefficient]

    So:
        E^1_{p,q} = Z^{ sum_{p-simplices sigma} C(rk(tau_0(sigma)), q) }

    Returns:
        e1 : dict (p, q) -> rank of E^1_{p,q} as a free abelian group
    """
    # Collect, for each p, all the bottom ranks of p-simplices
    # (with multiplicity — one entry per simplex)
    simplices_by_dim = defaultdict(list)
    for (n, bottom_rank), count in simplex_counts.items():
        simplices_by_dim[n].extend([bottom_rank] * count)

    if not simplices_by_dim:
        return {}

    max_p = max(simplices_by_dim.keys())
    max_r = max(r for ranks in simplices_by_dim.values() for r in ranks) if simplices_by_dim else 0

    e1 = {}
    for p in range(max_p + 1):
        for q in range(max_r + 1):
            rank = sum(comb(r, q) for r in simplices_by_dim.get(p, []))
            if rank > 0:
                e1[(p, q)] = rank

    return e1


# ── Printing ───────────────────────────────────────────────────────────────────

def print_vertex_summary(label, rank_counts, rank_essential):
    print(f"\n{'='*52}")
    print(f"  {label}")
    print(f"{'='*52}")
    print(f"  {'Rank':<8} {'Vertex Types':<16} {'Essential':<12}")
    print(f"  {'-'*38}")
    total_vt = total_ess = 0
    for rank in sorted(rank_counts):
        vt  = rank_counts[rank]
        ess = rank_essential.get(rank, 0)
        total_vt  += vt
        total_ess += ess
        print(f"  {rank:<8} {vt:<16} {ess:<12}")
    print(f"  {'-'*38}")
    print(f"  {'Total':<8} {total_vt:<16} {total_ess:<12}")
    print(f"{'='*52}")


def print_simplex_summary(simplex_counts):
    if not simplex_counts:
        print("  (no simplices)")
        return

    all_n = sorted(set(n for n, r in simplex_counts))
    all_r = sorted(set(r for n, r in simplex_counts))
    cw    = 10

    header = f"  {'dim n':<8}" + "".join(f"{'rank '+str(r):>{cw+2}}" for r in all_r) + f"{'Total':>{cw+2}}"
    sep    = "  " + "-" * (len(header) - 2)

    print(f"\n  Simplices  (rows = dimension, columns = bottom rank)")
    print(header)
    print(sep)

    grand_total = 0
    for n in all_n:
        row_total = 0
        row = f"  {str(n)+'-simpl':<8}"
        for r in all_r:
            val         = simplex_counts.get((n, r), 0)
            row_total  += val
            grand_total += val
            row += f"{val:>{cw+2}}"
        row += f"{row_total:>{cw+2}}"
        print(row)

    print(sep)
    foot = f"  {'Total':<8}"
    for r in all_r:
        foot += f"{sum(simplex_counts.get((n,r),0) for n in all_n):>{cw+2}}"
    foot += f"{grand_total:>{cw+2}}"
    print(foot)


def print_E1_page(e1):
    """
    Print the E^1 page as a grid.
    Rows    = q  (homological degree of stabiliser)
    Columns = p  (simplex dimension)
    Entry   = rank of E^1_{p,q} as free abelian group  (0 means trivial)

    Convention: q increases upward (as in a standard spectral sequence).
    """
    if not e1:
        print("  (E^1 page is trivial)")
        return

    all_p = sorted(set(p for p, q in e1))
    all_q = sorted(set(q for p, q in e1), reverse=True)  # q increases upward
    cw    = 8

    print(f"\n  E^1 page  (rows = q, columns = p, entry = rank of E^1_{{p,q}})")
    print(f"  Stab(sigma) = Z^{{rk(tau_0)}},  H_q(Z^r; Z) = Z^{{C(r,q)}}")

    header = f"  {'q \\ p':<6}" + "".join(f"{p:>{cw}}" for p in all_p)
    sep    = "  " + "-" * (len(header) - 2)
    print(header)
    print(sep)

    for q in all_q:
        row = f"  {q:<6}"
        for p in all_p:
            val  = e1.get((p, q), 0)
            row += f"{val:>{cw}}" if val > 0 else f"{'0':>{cw}}"
        print(row)

    print(sep)
    # Column totals
    foot = f"  {'total':<6}"
    for p in all_p:
        foot += f"{sum(e1.get((p,q),0) for q in set(q for _,q in e1)):>{cw}}"
    print(foot)


# ── Examples ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    examples = [
        ("4 vertices, no edges (Free group F_4)", [1, 2, 3, 4], []),
        ("5 vertices, one edge {1,2}",            [1, 2, 3, 4, 5], [(1, 2)]),
    ]

    for label, vertices, edges in examples:
        print(f"\n{'#'*60}")
        print(f"  {label}")
        print(f"{'#'*60}")

        valid_types, rc, re = compute_whitehead_poset(vertices, edges)
        print_vertex_summary(label, rc, re)

        print(f"\n  Computing simplices ({len(valid_types)} vertex types)...")
        sc = compute_simplices(valid_types)
        print_simplex_summary(sc)

        e1 = compute_E1_page(sc)
        print_E1_page(e1)

    # ── Add your own graph below ──────────────────────────────────────────────
    # vertices = [1, 2, 3, 4, 5]
    # edges    = [(1,2), (2,3), (3,4)]
    # valid_types, rc, re = compute_whitehead_poset(vertices, edges)
    # print_vertex_summary("My graph", rc, re)
    # sc = compute_simplices(valid_types)
    # print_simplex_summary(sc)
    # e1 = compute_E1_page(sc)
    # print_E1_page(e1)
