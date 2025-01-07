import editdistance
import re
import numpy as np

def clean_up_line(line):
    line = line.replace('"', ' ')
    line = re.sub(r'\s+', ' ', line)
    return line.strip()

def initialize_distance_matrix(strings):
    """
    Initializes a distance matrix for a list of strings based on edit distance with length ratio and distance thresholds.
    """
    n = len(strings)
    distance_matrix = np.full((n, n), float('inf'))  # Initialize the distance matrix with infinity
    
    for i in range(n):
        for j in range(i + 1, n):  # Only calculate the upper triangle
            len_i, len_j = len(strings[i]), len(strings[j])
            if 0.667 <= len_i / len_j <= 1.5:  # Check length ratio condition
                distance = editdistance.eval(strings[i], strings[j])
                if distance < (len_i + len_j) / 4:  # Apply distance threshold
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance  # Ensure symmetry

    return distance_matrix


def update_representative(cluster_ids, active, distance_matrix):
    """
    Updates cluster representatives by selecting the element with the smallest total distance to other members.
    """
    unique_clusters = np.unique(cluster_ids[active])

    for cluster in unique_clusters:
        members = np.where(cluster_ids == cluster)[0]
        
        if len(members) == 1:
            continue  # Skip clusters with a single member

        min_distance_sum = float('inf')
        best_representative = members[0]

        for member in members:
            distance_sum = np.sum(distance_matrix[member, members])
            if distance_sum < min_distance_sum:
                min_distance_sum = distance_sum
                best_representative = member

        active[members] = False
        active[best_representative] = True  # Activate the new representative

def cluster_tokens(strings, part_types, importance_scores):
    """
    Clusters strings based on edit distance, merging similar tokens and synchronizing type and score information.
    """
    n = len(strings)
    cluster_ids = np.arange(n)  # Initially, each string forms its own cluster
    distance_matrix = initialize_distance_matrix(strings)
    print(distance_matrix)
    active = np.ones(n, dtype=bool)  # Active status for cluster representatives

    # Perform hierarchical clustering
    while np.sum(active) > 1:
        min_distance = float('inf')
        x, y = -1, -1

        # Find the pair with the minimum distance
        for i in range(n):
            if not active[i]:
                continue
            for j in range(i + 1, n):
                if active[j] and distance_matrix[i][j] < min_distance:
                    min_distance = distance_matrix[i][j]
                    x, y = i, j

        if min_distance == float('inf'):
            break

        # Merge clusters and assign the smaller ID as the new cluster ID
        new_id = min(cluster_ids[x], cluster_ids[y])
        old_id = max(cluster_ids[x], cluster_ids[y])
        cluster_ids[cluster_ids == old_id] = new_id
        active[y] = False  # Deactivate the representative of the merged cluster

        # Update the cluster representatives
        update_representative(cluster_ids, active, distance_matrix)

    # Organize strings, types, and scores into their respective clusters
    final_clusters = {cid: [] for cid in set(cluster_ids)}
    type_clusters = {cid: [] for cid in set(cluster_ids)}
    score_clusters = {cid: [] for cid in set(cluster_ids)}

    for idx, cid in enumerate(cluster_ids):
        final_clusters[cid].append(strings[idx])
        type_clusters[cid].append(part_types[idx])
        score_clusters[cid].append(importance_scores[idx])

    # Consolidate type information for each cluster
    final_type_clusters = [list(set(types)) for types in type_clusters.values()]
    
    return list(final_clusters.values()), final_type_clusters


def longest_common_subsequence(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if s1[i - 1] == s2[j - 1]:
            lcs.append(s1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    lcs.reverse()

    # Insert spaces where characters are not adjacent in original strings
    spaced_lcs = []
    last_i, last_j = -1, -1  # Initialize to invalid indexes
    for char in lcs:
        current_i = s1.find(char, last_i + 1)
        current_j = s2.find(char, last_j + 1)
        if last_i != -1 and (current_i != last_i + 1 or current_j != last_j + 1):
            spaced_lcs.append(' ')
        spaced_lcs.append(char)
        last_i, last_j = current_i, current_j

    return ''.join(spaced_lcs)


def find_common_subsequence(str_list):
    """
    Finds the longest common subsequence among a list of strings.
    """
    if not str_list:
        return ""
    if len(str_list) == 1:
        return str_list[0]

    # Convert all strings in the list to lowercase
    str_list = [s.lower() for s in str_list]

    # Find the longest common subsequence between the first two strings
    common_sub = longest_common_subsequence(str_list[0], str_list[1])

    # Iteratively refine the common subsequence with remaining strings
    for s in str_list[2:]:
        common_sub = longest_common_subsequence(common_sub, s)
        if not common_sub:  # Stop early if no common subsequence remains
            break

    return common_sub


def decode_and_update_regex(expression):
    """
    Replaces all percent-encoded patterns in a string with a regex wildcard pattern.
    """
    # Find all patterns that match `%` followed by two hexadecimal digits
    encoded_parts = re.findall(r'%[0-9a-fA-F]{2}', expression)
    new_expression = expression

    for part in encoded_parts:
        new_expression = new_expression.replace(part, r'.*')  # Replace with regex wildcard

    return new_expression

def generate_signatures(clusters, type_clusters):
    """
    Generates regular expression signatures from each cluster's common subsequence.
    
    Args:
        clusters (list of list of str): List of string clusters.
        type_clusters (list of list of str): List of type information corresponding to each cluster.

    Returns:
        tuple: A tuple containing:
            - List of valid regex signatures.
            - List of types corresponding to valid clusters.
            - List of valid clusters that generated signatures.
    """
    signatures = []
    valid_types = []
    valid_clusters = []

    for idx, cluster in enumerate(clusters):
        # Find the longest common subsequence for the cluster
        common_subseq = find_common_subsequence(cluster)
        common_subseq = common_subseq.rstrip('\\')  # Remove trailing backslashes, if any
        common_subseq = clean_up_line(common_subseq)  # Clean up the line

        if common_subseq:
            # Replace all whitespace characters with '.*' to match any character sequence
            temp_signature = re.sub(r'[\s]+', r'.*', common_subseq)

            # Escape the string and preserve inserted '.*' sequences
            signature = re.escape(temp_signature).replace(r'\.\*', r'.*')
        else:
            # Use an empty string as the signature if no valid subsequence exists
            signature = ''

        # Check if the signature is valid and not overly generic
        if signature and signature not in ['/', '.*']:
            valid_types.append(type_clusters[idx])  # Save type information for valid clusters
            signatures.append(decode_and_update_regex(signature))  # Decode and finalize the signature
            valid_clusters.append(cluster)  # Add valid cluster to the result list

    return signatures, valid_types, valid_clusters


def generate_signatures(attack_strings, part_types, detailed_name, results_name1, results_name2, importance_scores, min_length):
    """
    Integrates clustering, finding common subsequences, and generating signatures.
    """
    # Perform clustering on attack strings
    clusters, type_clusters = cluster_tokens(attack_strings, part_types, importance_scores)
    print("Length of clusters:", len(clusters))
    print("Length of type_clusters:", len(type_clusters))
    
    # Generate signatures from clusters
    signatures, valid_types, clusters = generate_signatures(clusters, type_clusters)
    print("Length of signatures:", len(signatures))
    print("Length of valid_types:", len(valid_types))
    
    # Print cluster statistics
    cluster_sizes = [len(cluster) for cluster in clusters]
    print_cluster_statistics(cluster_sizes)
    
    # Save detailed cluster information and signatures
    save_cluster_details(clusters, signatures, valid_types, detailed_name, results_name1, results_name2, min_length=min_length)
    
    return signatures

def save_cluster_details(clusters, signatures, valid_types, filename, special_filename, normal_filename, min_size=10, min_length=10):
    """
    Saves detailed information about clusters and categorizes signatures into normal and special cases.
    """
    # Print lengths of inputs to verify alignment
    print("Length of clusters:", len(clusters))
    print("Length of signatures:", len(signatures))
    print("Length of valid_types:", len(valid_types))

    # Categorize special signatures
    short_signatures = []
    small_cluster_signatures = []
    substring_signatures = []

    # Sort clusters by size
    cluster_details = sorted(zip(clusters, signatures, valid_types), key=lambda x: len(x[0]))

    with open(filename, 'w') as file, open(special_filename, 'w') as special_file, open(normal_filename, 'w') as normal_file:
        for cluster, signature, types in cluster_details:
            # Write cluster size, types, and details
            file.write(f"Cluster size: {len(cluster)} - Types: {', '.join(types)}\n")
            for item in cluster:
                file.write(f"{item}\n")
            file.write(f"Generated Signature: {signature}\n\n" + "-" * 50 + "\n")

            # Calculate signature length after removing '.*'
            signature_length = len(re.sub(r'\.\*', '', signature))

            # Categorize based on signature properties
            if signature_length < min_length:
                short_signatures.append((signature, types))
                continue

            if any(sig != signature and signature in sig for sig in signatures):
                substring_signatures.append((signature, types))
                continue

            # Save normal signatures
            normal_file.write(f"{signature}\n")
            normal_file.write(f"Types: {', '.join(types)}\n")

        # Save special signatures
        if short_signatures:
            special_file.write("Short Signatures:\n")
            for sig, types in short_signatures:
                special_file.write(f"{sig}\n")
                special_file.write(f"Types: {', '.join(types)}\n")

        if small_cluster_signatures:
            special_file.write("Small Cluster Signatures:\n")
            for sig, types in small_cluster_signatures:
                special_file.write(f"{sig}\n")
                special_file.write(f"Types: {', '.join(types)}\n")

        if substring_signatures:
            special_file.write("Substring Signatures:\n")
            for sig, types in substring_signatures:
                special_file.write(f"{sig}\n")
                special_file.write(f"Types: {', '.join(types)}\n")


def print_cluster_statistics(cluster_sizes):
    """
    Prints statistics about cluster sizes by grouping them into predefined intervals.
    """
    from collections import Counter
    size_count = Counter(cluster_sizes)
    intervals = {}

    # Define intervals for cluster size
    ranges = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, float('inf'))]

    # Initialize interval counters
    for r in ranges:
        intervals[r] = 0

    # Count clusters in each interval
    for size in cluster_sizes:
        for r in ranges:
            if r[0] <= size <= r[1]:
                intervals[r] += 1
                break

    # Print results
    print("Cluster Size Statistics:")
    for r in ranges:
        if r[1] == float('inf'):
            print(f"{r[0]}+ : {intervals[r]}")
        else:
            print(f"{r[0]}-{r[1]} : {intervals[r]}")


def load_parameter_values(filepath):
    parameter_values = {}
    current_param = None
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            cleaned_line = clean_up_line(line)
            if cleaned_line.endswith(':'): 
                current_param = cleaned_line[:-1] 
                parameter_values[current_param] = []
            elif current_param and cleaned_line:
                parameter_values[current_param].append(cleaned_line)
    return parameter_values

def extract_abnormal_http_parts(file_path):
    """
    Extracts abnormal HTTP parts, their types, and importance scores from a file, 
    and filters parts based on a score threshold.
    """
    abnormal_http_parts = []
    part_types = []
    importance_scores = []

    # Temporary storage for parts being processed
    current_score = []
    current_part = []
    current_type = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Process "Important Token Scores"
            if line.startswith("Important Token Scores:"):
                current_score = []
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("Abnormal HTTP Parts:"):
                    current_score.append(eval(lines[i].strip()))
                    i += 1

            # Process "Abnormal HTTP Parts"
            if i < len(lines) and lines[i].strip().startswith("Abnormal HTTP Parts:"):
                current_part = []
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("Abnormal Parts types:"):
                    current_part.append(lines[i].rstrip('\n'))
                    i += 1

            # Process "Abnormal Parts types"
            if i < len(lines) and lines[i].strip().startswith("Abnormal Parts types:"):
                current_type = []
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("Evaluation Metrics"):
                    current_type.append(lines[i].strip())
                    i += 1

                # Skip the set if any component list is empty
                if not current_part or not current_type or not current_score:
                    i += 1
                    continue

                assert len(current_part) == len(current_type) == len(current_score), "Data mismatch in parts, types, or scores."

                # Process each abnormal part
                for part, ptype, score in zip(current_part, current_type, current_score):
                    if ptype in ["Body", "Query"]:
                        param_name = part.split('=')[0]
                        new_type = f"{ptype}:{param_name}"
                        part_value = part[len(param_name) + 1:]
                        score_trimmed = score[len(param_name) + 1:]  # Adjust score length
                        ptype = new_type
                        part = part_value
                        score = score_trimmed

                        if part:  # Only add if part is not empty
                            assert len(part) == len(score)
                            abnormal_http_parts.append(part)
                            part_types.append(ptype)
                            importance_scores.append(score)

                    elif ptype in ["Path"]:
                        abnormal_http_parts.append(part)
                        part_types.append(ptype)
                        importance_scores.append(score)
                    else:
                        abnormal_http_parts.append(part)
                        part_types.append("else")
                        importance_scores.append(score)
            else:
                i += 1
                continue

    # Filter parts based on score threshold
    score_threshold = 0
    filtered_parts = []
    filtered_part_types = []
    filtered_scores = []

    for part, ptype, scores in zip(abnormal_http_parts, part_types, importance_scores):
        # Filter characters below the threshold
        filtered_part = ''.join(char if score > score_threshold else ' ' for char, score in zip(part, scores))
        filtered_part = re.sub(r'\s+', ' ', filtered_part).strip()  # Compress spaces
        filtered_part_scores = [score if score > score_threshold else 0 for score in scores]

        if filtered_part:  # Only add non-empty parts
            filtered_parts.append(filtered_part)
            filtered_part_types.append(ptype)
            filtered_scores.append(filtered_part_scores)

    return filtered_parts, filtered_part_types, filtered_scores


def process_data_and_generate_signatures(input_file, detail_file, human_file, auto_file,min_length):
    abnormal_parts_list, part_types, importance_scores = extract_abnormal_http_parts(input_file)
    signatures = generate_signatures(abnormal_parts_list, part_types , detail_file, human_file, auto_file, importance_scores,min_length)

    return signatures


