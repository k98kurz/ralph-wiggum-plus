def stripws(fpath: str) -> int:
    try:
        with open(fpath, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        return 0

    count = 0
    new_lines = []
    for line in lines:
        stripped = line.rstrip()
        if len(stripped) < len(line.rstrip('\n')):
            count += 1
        new_lines.append(stripped + ('\n' if line.endswith('\n') else ''))

    if count > 0:
        with open(fpath, 'w') as f:
            f.writelines(new_lines)

    return count

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python stripws.py <file_path>")
        sys.exit(1)
    fpath = sys.argv[1]
    count = stripws(fpath)
    print(f"Stripped dangling whitespace from {count} lines.")
