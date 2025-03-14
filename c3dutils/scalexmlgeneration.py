from pathlib import Path
def convert_wsl_to_windows(wsl_path):
    # Check if the path starts with "/mnt/"
    if wsl_path.startswith("/mnt/"):
        # Extract the drive letter and file path
        drive_letter = wsl_path[5:6].upper()  # Get the drive letter (e.g., 'c' -> 'C')
        windows_path = wsl_path[6:].replace("/", "/")  # Convert slashes to backslashes
        # Combine to form the full Windows path
        return f"{drive_letter}:{windows_path}"
    else:
        # Return the original path if it's not a WSL path
        return wsl_path
def generatescale(trc):
    template_path="/mnt/c/OpenSim 4.4/Resources/Models/nature_paper_stroke_model/templates/scale_setup_run.xml"

    outputpath=trc.replace(".trc",".xml")
    import xml.etree.ElementTree as ET
    import xml.dom.minidom as minidom

    # Step 1: Create root
    tree = ET.parse(template_path)
    root = tree.getroot()
    
    generic_model_maker = root.find(".//GenericModelMaker")

    # Find the model_file element within GenericModelMaker
    model_file_element = generic_model_maker.find('model_file')

    # Update the model_file to a custom path
    
    model_file_element.text = 'C:/OpenSim 4.4/Resources/Models/nature_paper_stroke_model/09_03_model.osim' 
    
    model_scaler = root.find(".//ModelScaler")
    model_scaler.find('marker_file').text = convert_wsl_to_windows(trc).split("/")[-1]
    # Find the scaling_order element within ModelScaler
    scaling_order_element = model_scaler.find('scaling_order')

    # Update the scaling_order to only use manualScale
    scaling_order_element.text = 'measurements'
    marker_placer = root.find(".//MarkerPlacer")

    # Update the marker_file, coordinate_file, output_motion_file, output_model_file, output_marker_file paths
    marker_placer.find('marker_file').text = convert_wsl_to_windows(trc).split("/")[-1]  # Replace with custom path
    #marker_placer.find('coordinate_file').text = '/custom/path/to/coordinate_file.txt'  # Replace with custom path
    marker_placer.find('output_motion_file').text = convert_wsl_to_windows(trc).replace(".trc",".mot").split("/")[-1]  # Replace with custom path
    marker_placer.find('output_model_file').text = convert_wsl_to_windows(trc).replace(".trc",".osim").split("/")[-1]  # Replace with custom path
    #marker_placer.find('output_marker_file').text = '/custom/path/to/output_marker_set.xml'  # Replace with custom path

    tree.write(outputpath, encoding="utf-8", xml_declaration=True)
    print(f"XML file generated successfully: {outputpath}")


parent_folder = Path("/mnt/d/ubuntubackup/test/support_files/evaluation_mocaps/original/SOMA_manual_labeled")
reconstruct_folder = Path("/mnt/d/ubuntubackup/test/reconstructed/mosh_results/SOMA_manual_labeled")
original=[]
reconstructed=[]
for file in parent_folder.rglob("*.trc"):  # Use rglob to search recursively for .c3d files
    original.append(str(file))
for file in reconstruct_folder.rglob("*.trc"):  # Use rglob to search recursively for .c3d files
    reconstructed.append(str(file))


master=original+reconstructed
print(master)
for trc in master:
    generatescale(trc)