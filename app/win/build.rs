fn main() {
    println!("cargo:rerun-if-changed=nunchi.rc");
    // each icon, not the folder: rewriting a file in place does not change the folder's time
    for entry in std::fs::read_dir("icons").unwrap().flatten() {
        println!("cargo:rerun-if-changed={}", entry.path().display());
    }
    // the model built into the exe (main.rs include_bytes!)
    println!("cargo:rerun-if-changed=../../models/tr5n_102k.onnx");
    println!("cargo:rerun-if-changed=../../models/tr5n_102k.json");

    // the exe's details (Properties > Details, Task Manager): nunchi.rc plus a version block made from
    // Cargo.toml's version, so the version is written in one place
    let v = std::env::var("CARGO_PKG_VERSION").unwrap();
    let mut n: Vec<u16> = v.split(['.', '-']).filter_map(|p| p.parse().ok()).collect();
    n.resize(4, 0);
    let num = format!("{},{},{},{}", n[0], n[1], n[2], n[3]);
    let icons = std::fs::read_to_string("nunchi.rc").unwrap();
    let here = std::env::current_dir().unwrap();
    let icons = icons.replace("\"icons/", &format!("\"{}/icons/", here.display().to_string().replace('\\', "/")));
    let rc = format!(
        r#"{icons}
1 VERSIONINFO
FILEVERSION {num}
PRODUCTVERSION {num}
FILEOS 0x40004
FILETYPE 0x1
BEGIN
  BLOCK "StringFileInfo"
  BEGIN
    BLOCK "040904B0"
    BEGIN
      VALUE "CompanyName", "Minseok Kim"
      VALUE "FileDescription", "Nunchi - Korean/English auto switch"
      VALUE "FileVersion", "{v}"
      VALUE "InternalName", "nunchi"
      VALUE "LegalCopyright", "(c) 2026 Minseok Kim"
      VALUE "OriginalFilename", "nunchi.exe"
      VALUE "ProductName", "Nunchi"
      VALUE "ProductVersion", "{v}"
    END
  END
  BLOCK "VarFileInfo"
  BEGIN
    VALUE "Translation", 0x409, 1200
  END
END
"#
    );
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("nunchi_full.rc");
    std::fs::write(&out, rc).unwrap();
    embed_resource::compile(&out, embed_resource::NONE).manifest_optional().unwrap();

    // ONNX Runtime's prebuilt library links DirectML (GPU), which nunchi never uses: load it only if
    // something calls it, so the exe starts without DirectML.dll next to it
    println!("cargo:rustc-link-arg-bins=/DELAYLOAD:DirectML.dll");
    println!("cargo:rustc-link-lib=delayimp");
}
