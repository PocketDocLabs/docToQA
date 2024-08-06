def find_overlap(a, b):
    """Find the maximum overlap where suffix of 'a' is a prefix of 'b'."""
    max_overlap = 0
    overlap_text = ""
    # Check from full length of 'a' to just 1 character
    for i in range(1, min(len(a), len(b)) + 1):
        # Suffix of 'a' and prefix of 'b'
        if a[-i:] == b[:i]:
            max_overlap = i
            overlap_text = a[-i:]
    return max_overlap, overlap_text

def merge_strings(a, b):
    """Merge two strings with maximum overlap."""
    max_overlap, _ = find_overlap(a, b)
    if max_overlap > 0:
        return a + b[max_overlap:]
    return a + b  # No overlap, just concatenate

def merge_all_strings(strings):
    """Merge all strings in the list with healing overlaps."""
    while len(strings) > 1:
        best_a, best_b = None, None
        best_merged = None
        best_overlap = 0
        # Try to merge every pair to find the one with the maximum overlap
        for i in range(len(strings)):
            for j in range(len(strings)):
                if i != j:
                    a, b = strings[i], strings[j]
                    _, overlap_text = find_overlap(a, b)
                    if len(overlap_text) > best_overlap:
                        best_overlap = len(overlap_text)
                        best_a, best_b = i, j
                        best_merged = merge_strings(a, b)
        if best_merged:
            # Replace 'a' with the merged string and remove 'b'
            strings[best_a] = best_merged
            strings.pop(best_b)
        else:
            break  # No more merges possible
    return strings[0]


def merge_all_strings_advanced(strings):
    """Merge all strings in the list with healing overlaps using a more comprehensive approach."""
    changed = True
    while changed:
        changed = False
        n = len(strings)
        best_a, best_b = None, None
        best_merged = ""
        best_overlap = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    a, b = strings[i], strings[j]
                    overlap_len, _ = find_overlap(a, b)
                    if overlap_len > best_overlap:
                        best_overlap = overlap_len
                        best_a, best_b = i, j
                        best_merged = merge_strings(a, b)
                    # Check reverse direction as well
                    overlap_len, _ = find_overlap(b, a)
                    if overlap_len > best_overlap:
                        best_overlap = overlap_len
                        best_a, best_b = j, i
                        best_merged = merge_strings(b, a)

        if best_a is not None and best_b is not None:
            strings[best_a] = best_merged
            strings.pop(best_b)
            changed = True  # Indicate a successful merge to continue the loop

    # When no more merges are possible, strings should be down to 1
    if len(strings) == 1:
        return strings[0]
    else:
        # If multiple unmergeable strings remain, join them as a fallback
        return '\n\n'.join(strings)
