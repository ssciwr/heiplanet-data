from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent
script_folder = "heiplanet_data"
src = root / script_folder

for path in sorted(src.glob("*.py")):
    module_path = path.relative_to(root).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        if not parts or parts[-1] == script_folder:
            continue  # no md file for the main folder
        doc_path = doc_path.with_name("index.md")  # bind page init to section
        full_doc_path = full_doc_path.with_name("index.md")
        nav[parts] = doc_path.as_posix()
    elif parts[-1] == "__main__":
        continue
    else:
        nav[parts] = doc_path.as_posix()

    # write to virtual md files
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        # attach html class named moduleh1
        fd.write(f"# {ident} module" + " {: .moduleh1 }\n\n")
        fd.write(f"::: {ident}\n")

    # set edit_uri
    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

# write the nav to md file
with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
