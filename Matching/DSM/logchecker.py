import re

def parse_log_file(log_filename):
    print("Checking logs for greedy_auto_label, greedy_matcher, and nonperish_bidirectional_GAL.")

    matches = {
        "greedy_auto_label": set(),
        "greedy_matcher": set(),
        "nonperish_bidirectional_GAL": set()
    }
    abandonments = {
        "greedy_auto_label": set(),
        "greedy_matcher": set(),
        "nonperish_bidirectional_GAL": set()
    }
    match_count = {
        "greedy_auto_label": 0,
        "greedy_matcher": 0,
        "nonperish_bidirectional_GAL": 0
    }
    current_matcher = None

    with open(log_filename, 'r') as file:
        for line in file:
            # Detect when a new matcher phase starts
            if "Starting greedy_matcher" in line:
                current_matcher = "greedy_matcher"
                print("Entered greedy_matcher phase.")
                continue
            elif "Starting greedy_auto_label" in line:
                current_matcher = "greedy_auto_label"
                print("Entered greedy_auto_label phase.")
                continue
            elif "Starting nonperish_bidirectional_GAL" in line:
                current_matcher = "nonperish_bidirectional_GAL"
                print("Entered nonperish_bidirectional_GAL phase.")
                continue

            # Ignore lines if no matcher is active
            if current_matcher is None:
                continue

            # Match detection
            match = re.search(r'Matched Driver ([0-9\.]+) \| Location: .* with Rider ([0-9\.]+)', line)
            match_alt = re.search(r'Matched Rider ([0-9\.]+) \| Location: .* with Driver ([0-9\.]+)', line)

            if match or match_alt:
                driver_time, rider_time = match.groups() if match else match_alt.groups()
                if (driver_time, rider_time) in matches[current_matcher]:
                    print(f"[ERROR] Duplicate match detected in {current_matcher}: Driver {driver_time} and Rider {rider_time}.")
                else:
                    matches[current_matcher].add((driver_time, rider_time))
                    match_count[current_matcher] += 1

            # Abandonment detection
            abandon_match = re.search(r'(Driver|Rider) ([0-9\.]+) abandoned', line)
            if abandon_match:
                entity_type, entity_time = abandon_match.groups()
                if any(entity_time in pair for pair in matches[current_matcher]):
                    print(f"[ERROR] {entity_type} {entity_time} abandoned after being matched in {current_matcher}. Possible logic error.")
                abandonments[current_matcher].add(entity_time)

    # Print Summary Report
    print("\nSummary Report:")
    for matcher in match_count:
        print(f"Total Matches in {matcher}: {match_count[matcher]}")
        print(f"Total Abandonments in {matcher}: {len(abandonments[matcher])}")

    print("\nSanity Check Completed.")

if __name__ == "__main__":
    parse_log_file("logs.txt")
