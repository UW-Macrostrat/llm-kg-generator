import json

counter = 0


def dump_output(output: str, file_path: str = "out/output") -> None:
    global counter
    counter += 1
    if counter > 10:
        with open(file_path, "w") as file:
            file.write("")
        counter = 0

    json_obj = json.loads(output)

    with open(file_path, "a") as file:
        pretty_json = json.dumps(json_obj, indent=4)
        file.write(pretty_json)
        file.write("\n")
