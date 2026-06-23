# ABOUTME: Molecular graph analysis for lipid head/tail detection
# ABOUTME: Classifies atoms into head groups and tails using graph algorithms on SMILES
"""Molecular graph analysis for lipid head/tail detection."""

import operator
from collections import deque

import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw


def check_branch_points_not_in_cycles(mol, branch_point_indices, exclude_elements=[]):
    """Filter branch points to those not part of any ring.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule object.
    branch_point_indices : list of int
        Atom indices representing branch points.
    exclude_elements : list of str, optional
        Element symbols to keep even if they are in a cycle.

    Returns
    -------
    list of int
        Branch point indices that are not in a cycle (or whose element
        is in `exclude_elements`).

    """
    # Get the smallest set of smallest rings (SSSR)
    rings = Chem.GetSymmSSSR(mol)

    elements = {i: mol.GetAtomWithIdx(i).GetSymbol() for i in branch_point_indices}

    # Create a set of all atoms that are part of any cycle
    cycle_atoms = set()
    for ring in rings:
        cycle_atoms.update(ring)

    # Collect branch points that are not in a cycle
    not_in_cycle_indices = [
        idx
        for idx in branch_point_indices
        if idx not in cycle_atoms or elements[idx] in exclude_elements
    ]

    return not_in_cycle_indices


def remove_leaves(graph):
    """Remove leaf nodes (nodes with only one connection) from the graph."""
    graph = graph.copy()
    leaves = [node for node in graph.nodes if graph.degree[node] == 1]
    graph.remove_nodes_from(leaves)
    return graph


def detect_lipid_parts_from_smiles_modified(smiles, head_search_radius=3, min_tail_distance=6):
    """Detect head group, tail termini, and branch points of a lipid from SMILES.

    Parse the molecular graph, trim terminal atoms, identify branch points,
    and classify endpoints as head group or tail atoms.

    Parameters
    ----------
    smiles : str
        SMILES string of the lipid molecule.
    head_search_radius : int, optional
        Maximum graph distance to search for head-group heteroatoms from
        trimmed endpoints. Default is 3.
    min_tail_distance : int, optional
        Minimum graph distance from the head-group branch point required for
        a valid tail endpoint. Default is 6.

    Returns
    -------
    head_index : int or None
        Atom index of the detected head group, or None if parsing fails.
    true_tail_indices : list of int or None
        Atom indices of the detected tail termini, or None if parsing fails.
    branch_indices : list of int or None
        Atom indices of branch points, or None if parsing fails.

    """
    # Load molecule from SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Could not load molecule from SMILES: {smiles}")
        return None, None, None

    # Get full adjacency matrix
    full_adjacency_matrix = Chem.GetAdjacencyMatrix(mol)
    graph = nx.from_numpy_array(full_adjacency_matrix)

    # shave off 2 layers of terminal atoms
    graph_trimmed = remove_leaves(remove_leaves(graph))
    new_endpoints = [n for n in graph_trimmed.nodes if graph_trimmed.degree[n] == 1]
    branch_indices = [n for n in graph_trimmed.nodes if graph_trimmed.degree[n] > 2]

    # check if branch points are part of a cyclic structure
    if len(branch_indices) > 1:
        branch_indices = check_branch_points_not_in_cycles(mol, branch_indices)

    # Classify endpoints
    head_index, tail_indices = classify_endpoints(
        mol,
        graph,
        new_endpoints,
        branch_indices,
        head_search_radius,
        min_tail_distance,
    )

    # Map tail indices to original terminal atoms
    true_tail_indices = map_to_original_terminals(graph, tail_indices)

    if head_index in true_tail_indices:
        true_tail_indices.remove(head_index)
    if head_index in branch_indices:
        branch_indices.remove(head_index)

    # filter out non-carbon tail indices
    true_tail_indices = [
        idx for idx in true_tail_indices if mol.GetAtomWithIdx(idx).GetSymbol() == "C"
    ]

    return head_index, true_tail_indices, branch_indices


def classify_endpoints(
    mol, graph, endpoints, branch_indices, head_search_radius, min_tail_distance=10
):
    """Classify trimmed-graph endpoints as head group or tail atoms.

    Search for the head group by looking for terminal hydroxyl, nearby nitrogen,
    farthest nitrogen, farthest oxygen, or any heteroatom (in that priority order).
    Remaining endpoints that pass distance and element filters become tails.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule object.
    graph : networkx.Graph
        Full molecular connectivity graph.
    endpoints : list of int
        Endpoint atom indices from the twice-trimmed graph.
    branch_indices : list of int
        Atom indices of non-cyclic branch points.
    head_search_radius : int
        Maximum graph distance to search for head-group heteroatoms.
    min_tail_distance : int, optional
        Minimum graph distance from the head-group branch point for a valid
        tail. Default is 10.

    Returns
    -------
    head_index : int or None
        Atom index of the detected head group.
    tail_indices : list of int
        Atom indices classified as tail endpoints.

    """
    head_index = None
    tail_indices = []

    if nx.number_connected_components(graph) > 1:
        raise ValueError("Molecule graph is not fully connected.")
    elements = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]

    # Check all atoms in the molecule for terminal hydroxyl groups
    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)
        # Check if the oxygen is terminal (only one bond)
        if atom.GetSymbol() == "O" and graph.degree[atom_idx] == 1:
            neighbors = tuple(nx.all_neighbors(graph, atom_idx))
            neighbor_idx = int(neighbors[0])
            neighbor_atom = mol.GetAtomWithIdx(neighbor_idx)
            if neighbor_atom.GetSymbol() == "C":
                # Check if the oxygen is connected to a hydrogen
                if atom.GetTotalNumHs() > 0:  # Ensure it's a hydroxyl (–OH)
                    head_index = atom_idx  # Assign the hydroxyl oxygen as headgroup
                    break

    # If no terminal hydroxyl found, check for terminal nitrogen
    if head_index is None:
        for endpoint in endpoints:
            # nitrogen_result = search_for_element(
            #     mol, full_adjacency_matrix, endpoint, "N", head_search_radius
            # )
            paths = {
                t: p
                for t, p in nx.single_source_shortest_path(
                    graph, endpoint, cutoff=head_search_radius - 1
                ).items()
                if elements[t] == "N" and t not in branch_indices
            }
            # if nitrogen_result is not None and nitrogen_result not in branch_indices:
            #     print(endpoint, paths, nitrogen_result)
            #     assert len(paths) > 0
            if paths:
                head_index = next(iter(paths))
                # assert nitrogen_result == head_index
                break
            # if nitrogen_result is not None:
            #     # Ensure it's not a branch point
            #     if nitrogen_result not in branch_indices:
            #         head_index = nitrogen_result  # Ensure we return the nitrogen itself
            #         break  # Stop if we find a terminal nitrogen

    # If no terminal nitrogen found, check for the farthest nitrogen
    if head_index is None:
        all_path_lengths = {}
        for endpoint in endpoints:
            # farthest_nitrogen = search_for_farthest_element(
            #     mol, full_adjacency_matrix, endpoint, "N"
            # )
            # if farthest_nitrogen is not None:
            #     # Ensure it's not a branch point
            #     if farthest_nitrogen not in branch_indices:
            #         head_index = farthest_nitrogen
            #         break  # Stop if we find a farthest nitrogen
            paths = {
                t: p
                for t, p in nx.single_source_shortest_path(graph, endpoint).items()
                if elements[t] == "N" and t != endpoint and t not in branch_indices
            }
            path_lengths = {t: len(p) - 1 for t, p in paths.items()}
            all_path_lengths[endpoint] = path_lengths
        if all_path_lengths:
            from collections import defaultdict

            sum_path_lengths = defaultdict(int)
            for lengths in all_path_lengths.values():
                for t, length in lengths.items():
                    sum_path_lengths[t] += length
            if sum_path_lengths:
                head_index = max(sum_path_lengths, key=sum_path_lengths.get)

    # If no nitrogen found, check for oxygen in the endpoints
    if head_index is None:
        all_path_lengths = {}
        for endpoint in endpoints:
            # oxygen_result = search_for_farthest_element(mol, full_adjacency_matrix, endpoint, "O")
            # if oxygen_result is not None:
            #     # Ensure it's not a branch point
            #     if oxygen_result not in branch_indices:
            #         head_index = oxygen_result  # Ensure we return the oxygen itself
            #         break
            paths = {
                t: p
                for t, p in nx.single_source_shortest_path(graph, endpoint).items()
                if elements[t] == "O" and t != endpoint and t not in branch_indices
            }
            path_lengths = {t: len(p) - 1 for t, p in paths.items()}
            all_path_lengths[endpoint] = path_lengths
        if all_path_lengths:
            from collections import defaultdict

            sum_path_lengths = defaultdict(int)
            for lengths in all_path_lengths.values():
                for t, length in lengths.items():
                    sum_path_lengths[t] += length
            if sum_path_lengths:
                head_index = max(sum_path_lengths, key=sum_path_lengths.get)

    # If no headgroup found, search for the farthest non-carbon heteroatom
    if head_index is None:
        for endpoint in endpoints:
            path_lengths = {
                t: p
                for t, p in nx.single_source_shortest_path_length(graph, endpoint).items()
                if elements[t] in ["N", "O", "S", "P"]
            }
            farthest_heteroatom = max(path_lengths, key=path_lengths.get) if path_lengths else None
            # farthest_heteroatom = search_for_farthest_element(
            #     mol,
            #     full_adjacency_matrix,
            #     endpoint,
            #     "N",
            #     heteroatoms={"N", "O", "S", "P"},
            # )
            if farthest_heteroatom is not None:
                head_index = farthest_heteroatom
                break

    # Identify carbons adjacent to the headgroup (if it's a nitrogen)
    headgroup_adjacent_carbons = set()
    if head_index is not None and mol.GetAtomWithIdx(head_index).GetSymbol() == "N":
        neighbors = nx.all_neighbors(graph, head_index)
        for neighbor in neighbors:
            if mol.GetAtomWithIdx(int(neighbor)).GetSymbol() == "C":
                headgroup_adjacent_carbons.add(int(neighbor))

    # determine head-group branch index
    head_branch_index = None
    if branch_indices:
        head_branch_path_lengths = [
            nx.shortest_path_length(graph, bi, head_index) for bi in branch_indices
        ]
        head_branch_index = branch_indices[np.argmin(head_branch_path_lengths)]

    # Classify remaining endpoints as tails, excluding headgroup-adjacent carbons and checking
    # minimum distance from branch points
    true_endpoints = map_to_original_terminals(graph, endpoints)
    for endpoint, true_endpoint in zip(endpoints, true_endpoints):
        if true_endpoint != head_index and endpoint not in headgroup_adjacent_carbons:
            is_valid_tail = True

            if elements[true_endpoint] != "C":
                is_valid_tail = False

            # tail needs to be at least min_tail_distance away from head atom's branch point
            if head_branch_index is not None:
                distance = nx.shortest_path_length(graph, endpoint, head_branch_index)
                if distance < min_tail_distance:
                    is_valid_tail = False

                distance_to_head = nx.shortest_path_length(graph, true_endpoint, head_index)
                if distance_to_head < min_tail_distance:
                    is_valid_tail = False
                # also, the tail cannot be in the subgraph of the head's branch point and the head
                # G_copy = G.copy()
                # paths_to_head = nx.all_shortest_paths(G, head_branch_index, head_index)
                # removed = []
                # for path_to_head in paths_to_head:
                #     edge = (path_to_head[0], path_to_head[1])
                #     if edge not in removed:
                #         removed.append(edge)
                #         G_copy.remove_edge(path_to_head[0], path_to_head[1])
                # components = list(nx.connected_components(G_copy))
                # subgraphs = [G_copy.subgraph(component).copy() for component in components]
                # if len(subgraphs) != 2:
                #     raise ValueError(
                #         "Expected exactly two subgraphs after removing head-branch edge."
                #     )
                # for subgraph in subgraphs:
                #     if head_index in subgraph.nodes and endpoint in subgraph.nodes:
                #         is_valid_tail = False
                #         break

            if is_valid_tail:
                tail_indices.append(endpoint)

    # Remove head_index from tail_indices if it exists
    if head_index in tail_indices:
        tail_indices.remove(head_index)

    # if no tail indices found, try to find a contiguous carbon chain
    if len(tail_indices) == 0 and head_index is not None:
        import warnings

        warnings.warn("Falling back to idiotic chemist rule.")
        for endpoint in endpoints:
            if endpoint != head_index and endpoint not in headgroup_adjacent_carbons:
                # find the closest branch point to endpoint
                idx = np.argmin(
                    [
                        nx.shortest_path_length(graph, endpoint, branch_idx)
                        for branch_idx in branch_indices
                    ]
                )
                closest_branch_point = branch_indices[idx]
                path = nx.shortest_path(graph, endpoint, closest_branch_point)
                elements_along_path = operator.itemgetter(*path)(elements)
                if all(ele == "C" for ele in elements_along_path):
                    tail_indices.append(endpoint)

    return head_index, tail_indices


def find_closest_node_single_source(graph, source_node, candidate_nodes, cutoff=2):
    """Find the closest candidate node to a source using shortest-path distance.

    Parameters
    ----------
    graph : networkx.Graph
        Molecular connectivity graph.
    source_node : int
        Starting node index.
    candidate_nodes : list of int
        Node indices to consider as targets.
    cutoff : int, optional
        Maximum path length to search. Default is 2.

    Returns
    -------
    closest_node : int or None
        Index of the closest candidate, or None if none reachable.
    min_distance : int or float
        Shortest-path distance to the closest candidate (inf if none found).

    """
    # Compute shortest paths from source to all nodes
    distances = nx.single_source_shortest_path_length(graph, source_node, cutoff=cutoff)

    # Find the closest candidate node
    min_distance = float("inf")
    closest_node = None

    for candidate in candidate_nodes:
        if candidate in distances and distances[candidate] < min_distance:
            min_distance = distances[candidate]
            closest_node = candidate

    return closest_node, min_distance


def map_to_original_terminals(graph, tail_indices):
    """Map trimmed-graph endpoints back to the nearest original terminal atoms.

    Parameters
    ----------
    graph : networkx.Graph
        Full molecular connectivity graph (before trimming).
    tail_indices : list of int
        Atom indices from the trimmed graph to map back.

    Returns
    -------
    list of int
        Corresponding terminal atom indices in the original graph.

    """
    # G = nx.from_numpy_array(full_adjacency_matrix)
    terminal_atoms = [node for node in graph if graph.degree[node] == 1]

    # cutoff=2 because 2 layers were shaved off
    true_tail_indices = [
        find_closest_node_single_source(graph, tail, terminal_atoms, cutoff=2)[0]
        for tail in tail_indices
    ]
    assert len(set(true_tail_indices)) == len(true_tail_indices), (
        "Duplicate true tail indices found."
    )
    return true_tail_indices


def remove_duplicates_and_sort(connections):
    """Deduplicate and sort a list of atom-index pairs.

    Parameters
    ----------
    connections : list of list of int
        Pairs (or sequences) of atom indices, possibly with duplicates or
        reversed ordering.

    Returns
    -------
    list of list of int
        Unique connections, each internally sorted.

    """
    # Use a set to store unique sorted lists
    unique_connections = set(tuple(sorted(conn)) for conn in connections)

    # Convert back to a list of lists
    return [list(conn) for conn in unique_connections]


def analyze_molecular_graph(mol, headgroup_index, tail_indices, branch_indices):
    """Partition a molecular graph into segments between key structural points.

    Trace paths between head group, tail termini, and branch points via BFS,
    then assign every remaining atom to the nearest segment.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule object.
    headgroup_index : int
        Atom index of the head group.
    tail_indices : list of int
        Atom indices of tail termini.
    branch_indices : list of int
        Atom indices of branch points.

    Returns
    -------
    list of list of int
        Segments of atom indices, each sorted, covering the full molecule.

    """
    # Combine all points of interest
    points_of_interest = set([headgroup_index] + tail_indices + branch_indices)

    # Get the adjacency list representation of the molecule
    adjacency_list = [[] for _ in range(mol.GetNumAtoms())]
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency_list[i].append(j)
        adjacency_list[j].append(i)

    connections = []

    # Process tail atoms
    for tail in tail_indices:
        connections.extend(find_connections_from_point(adjacency_list, tail, points_of_interest))

    # Process branch points
    for branch_point in branch_indices:
        connections.extend(
            find_connections_from_point(adjacency_list, branch_point, points_of_interest)
        )

    connections = remove_duplicates_and_sort(connections)

    # Assign adjacent atoms
    connections = assign_adjacent_atoms(mol, connections, points_of_interest)
    connections = [sorted(sublist) for sublist in connections]

    return connections


def find_connections_from_point(adjacency_list, start, points_of_interest):
    """Find all BFS paths from a start atom to other points of interest.

    Parameters
    ----------
    adjacency_list : list of list of int
        Per-atom neighbor lists for the molecule.
    start : int
        Starting atom index (must be in `points_of_interest`).
    points_of_interest : set of int
        Atom indices at which to terminate paths.

    Returns
    -------
    list of list of int
        Each entry is a path (list of atom indices) from `start` to another
        point of interest.

    """
    paths = []  # Initialize a list to store paths
    queue = deque([(start, [start])])
    visited = set([start])

    while queue:
        current, path = queue.popleft()

        for neighbor in adjacency_list[current]:
            if neighbor in points_of_interest and neighbor != start:
                paths.append(path + [neighbor])  # Append the current path to the paths list
            elif neighbor not in visited and neighbor not in points_of_interest:
                new_path = path + [neighbor]
                queue.append((neighbor, new_path))
                visited.add(neighbor)

    # print(paths)  # Print the paths for debugging
    return paths  # Return only the paths


def assign_adjacent_atoms(mol, segments, points_of_interest):
    """Assign unassigned atoms to the nearest existing segment.

    Atoms not already in any segment and not in `points_of_interest` are
    appended to whichever segment contains their closest neighbor by
    shortest-path distance.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule object.
    segments : list of list of int
        Existing segments of atom indices (modified in place).
    points_of_interest : set of int
        Head, tail, and branch atom indices (excluded from assignment).

    Returns
    -------
    list of list of int
        The input segments with unassigned atoms appended.

    """
    # Check if segments is empty to avoid concatenation error
    if not segments:
        assigned_atoms = set()
    else:
        # Flatten the segments to check which atoms are already assigned
        assigned_atoms = set(np.concatenate(segments))

    G = nx.from_numpy_array(Chem.GetAdjacencyMatrix(mol))

    # Assign unassigned atoms to the nearest segment based on connectivity
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if idx in points_of_interest or idx in assigned_atoms:
            continue  # Skip if the atom is a point of interest or already assigned

        # Initialize variables to find the closest segment
        closest_segment = None
        min_distance = float("inf")

        # Check distance to each segment
        for segment in segments:
            for seg_atom in segment:
                # Calculate distance
                test_distance = nx.shortest_path_length(G, idx, seg_atom)
                # assert test_distance == test_distance2, f"{test_distance} != {test_distance2}"
                if test_distance < min_distance:
                    min_distance = test_distance
                    closest_segment = segment

        # If a closest segment is found, append the unassigned atom to it
        if closest_segment is not None:
            closest_segment.append(idx)  # Add the unassigned atom to the segment
            assigned_atoms.add(idx)  # Mark this atom as assigned

    return segments  # Return the updated segments list


def visualize_lipid_parts_from_smiles(smiles, output_file="lipid_parts.png"):
    """Visualize detected lipid head and tail groups from SMILES.

    Render a 2D image of the molecule with head (red), tail (blue), and
    branch (green) atoms highlighted. Intended for use in a Jupyter notebook.

    Parameters
    ----------
    smiles : str
        SMILES string of the lipid molecule.
    output_file : str, optional
        Path for the output PNG image. Default is ``"lipid_parts.png"``.

    Returns
    -------
    tuple of (rdkit.Chem.Mol, int, list of int, list of int) or None
        ``(mol, head_index, tail_indices, branch_indices)`` on success,
        or None if the molecule cannot be parsed.

    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Could not load molecule from SMILES: {smiles}")
        return

    head_index, tail_indices, branch_indices = detect_lipid_parts_from_smiles_modified(smiles)

    # Create atom highlighting
    highlight_atoms = {}

    # Color head group
    if head_index is not None:
        highlight_atoms[int(head_index)] = (1, 0, 0)  # Red for head group

    # Color tail groups
    for idx in tail_indices:
        highlight_atoms[int(idx)] = (0, 0, 1)  # Blue for tail groups

    # Color branch points
    for idx in branch_indices:
        highlight_atoms[int(idx)] = (0, 1, 0)  # Green for branch points

    # Color remaining atoms
    for atom_idx in range(mol.GetNumAtoms()):
        if atom_idx not in highlight_atoms:
            highlight_atoms[atom_idx] = (0.8, 0.8, 0.8)  # Gray for neutral atoms

    # Generate 2D coordinates if they don't exist
    if not mol.GetNumConformers():
        AllChem.Compute2DCoords(mol)

    # Draw molecule with highlights
    drawer = Draw.MolDraw2DCairo(800 * 2, 600 * 2)
    drawer.drawOptions().addAtomIndices = True
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(highlight_atoms.keys()),
        highlightAtomColors=highlight_atoms,
    )
    drawer.FinishDrawing()
    with open(output_file, "wb") as f:
        f.write(drawer.GetDrawingText())

    print(f"Visualization saved to {output_file}")
    return mol, head_index, tail_indices, branch_indices


def create_lipid_assignment(
    mol,
    head_index,
    tail_indices,
    branch_indices,
    output_file="lipid_assignment.png",
):
    """Create a color-coded visualization of lipid segment assignments.

    Partition the molecule into segments between head, tail, and branch
    points, then render each segment in a distinct color.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule object.
    head_index : int
        Atom index of the head group.
    tail_indices : list of int
        Atom indices of tail termini.
    branch_indices : list of int
        Atom indices of branch points.
    output_file : str, optional
        Path for the output PNG image. Default is ``"lipid_assignment.png"``.

    Returns
    -------
    list of list of int
        Segments of atom indices used in the visualization.

    """
    # Generate 2D coordinates if they don't exist
    if not mol.GetNumConformers():
        AllChem.Compute2DCoords(mol)

    # Create atom highlighting
    highlight_atoms = {}

    # Color head group
    if head_index is not None:
        highlight_atoms[int(head_index)] = (1, 0, 0)  # Red for head group

    # Color tail groups
    for idx in tail_indices:
        highlight_atoms[int(idx)] = (0, 0, 1)  # Blue for tail groups

    # Color branch points
    for idx in branch_indices:
        highlight_atoms[int(idx)] = (0, 1, 0)  # Green for branch points

    # Assign distinct colors to each segment
    segments = analyze_molecular_graph(mol, head_index, tail_indices, branch_indices)
    distinct_colors = [
        (1, 0.5, 0),  # Orange
        (0.5, 0, 1),  # Purple
        (0, 1, 1),  # Cyan
        (1, 0, 0.5),  # Magenta
        (0.5, 0.5, 0),  # Olive
        (1, 1, 0),  # Yellow
        (0, 0.5, 0),  # Dark Green
        (1, 0, 0),  # Red
        (0, 0, 1),  # Blue
        (0.5, 0.5, 1),  # Light Blue
    ]

    for i, seg in enumerate(segments):
        segment = np.sort(seg).tolist()
        color = distinct_colors[i % len(distinct_colors)]  # Cycle through distinct colors
        for atom_idx in segment:
            highlight_atoms[atom_idx] = color  # Assign color to all atoms in the segment

    # Draw molecule with highlights
    drawer = Draw.MolDraw2DCairo(800, 600)
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(highlight_atoms.keys()),
        highlightAtomColors=highlight_atoms,
    )
    drawer.FinishDrawing()
    with open(output_file, "wb") as f:
        f.write(drawer.GetDrawingText())

    print(f"Lipid assignment visualization saved to {output_file}")
    return segments


def smiles_substructure_match(query_smiles: str, target_smiles: str) -> bool:
    """Check if target molecule contains query as a substructure.

    Parameters
    ----------
    query_smiles : str
        SMILES of the query substructure.
    target_smiles : str
        SMILES of the target molecule.

    Returns
    -------
    bool
        True if target contains query substructure.

    Raises
    ------
    ValueError
        If either SMILES string is invalid.

    """
    from rdkit import RDLogger

    # Suppress RDKit stderr spam for invalid SMILES (expected input case)
    RDLogger.DisableLog("rdApp.*")
    try:
        query_mol = Chem.MolFromSmiles(query_smiles)
        if query_mol is None:
            raise ValueError(f"Invalid query SMILES: {query_smiles}")
        target_mol = Chem.MolFromSmiles(target_smiles)
        if target_mol is None:
            raise ValueError(f"Invalid target SMILES: {target_smiles}")
        return target_mol.HasSubstructMatch(query_mol)
    finally:
        RDLogger.EnableLog("rdApp.*")
