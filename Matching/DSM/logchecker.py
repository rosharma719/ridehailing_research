import re
from collections import defaultdict

def parse_log_file(log_filename):
    print("Checking logs for greedy_auto_label, greedy_matcher, and nonperish_bidirectional_GAL.")

    matchers = ["greedy_auto_label", "greedy_matcher", "nonperish_bidirectional_GAL"]

    matches = {matcher: set() for matcher in matchers}
    abandonments = {matcher: set() for matcher in matchers}
    match_count = {matcher: 0 for matcher in matchers}

    rider_driver_counts = {matcher: defaultdict(lambda: defaultdict(int)) for matcher in matchers}
    driver_rider_counts = {matcher: defaultdict(lambda: defaultdict(int)) for matcher in matchers}

    current_matcher = None

    with open(log_filename, 'r') as file:
        for line in file:
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

            if current_matcher is None:
                continue

            # Abandonment parsing first and independently
            abandon_match = re.search(r'(Driver|Rider) ([0-9.]+) abandoned', line)
            if abandon_match:
                entity_type, entity_time = abandon_match.groups()
                if any(entity_time in pair for pair in matches[current_matcher]):
                    print(f"[ERROR] {entity_type} {entity_time} abandoned after being matched in {current_matcher}. Possible logic error.")
                abandonments[current_matcher].add(entity_time)

            # Match detection
            match = re.search(r'Matched Driver ([0-9.]+) \| Location: (\d+) with Rider ([0-9.]+) \| Location: (\d+)', line)
            match_alt = re.search(r'Matched Rider ([0-9.]+) \| Location: (\d+) with Driver ([0-9.]+) \| Location: (\d+)', line)

            if match:
                driver_time, driver_loc, rider_time, rider_loc = match.groups()
            elif match_alt:
                rider_time, rider_loc, driver_time, driver_loc = match_alt.groups()
            else:
                continue

            key = (driver_time, rider_time)
            if key in matches[current_matcher]:
                print(f"[ERROR] Duplicate match detected in {current_matcher}: Driver {driver_time} and Rider {rider_time}.")
            else:
                matches[current_matcher].add(key)
                match_count[current_matcher] += 1
                rider_driver_counts[current_matcher][int(rider_loc)][int(driver_loc)] += 1
                driver_rider_counts[current_matcher][int(driver_loc)][int(rider_loc)] += 1

    print("\nSummary Report:")
    for matcher in matchers:
        print(f"Total Matches in {matcher}: {match_count[matcher]}")
        print(f"Total Abandonments in {matcher}: {len(abandonments[matcher])}")

    print("\nMatch Frequencies by Node:")
    for matcher in matchers:
        print(f"\n[{matcher}] Rider Node → Driver Node Match Counts:")
        for rider in sorted(rider_driver_counts[matcher]):
            for driver in sorted(rider_driver_counts[matcher][rider]):
                count = rider_driver_counts[matcher][rider][driver]
                print(f"Rider {rider} → Driver {driver}: {count} matches")

        print(f"\n[{matcher}] Driver Node → Rider Node Match Counts:")
        for driver in sorted(driver_rider_counts[matcher]):
            for rider in sorted(driver_rider_counts[matcher][driver]):
                count = driver_rider_counts[matcher][driver][rider]
                print(f"Driver {driver} → Rider {rider}: {count} matches")

    print("\nSanity Check Completed.")

if __name__ == "__main__":
    parse_log_file("logs.txt")
