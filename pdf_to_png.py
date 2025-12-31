from pdf2image import convert_from_path
import os
from pathlib import Path

def extract_first_page_as_png(pdf_path, output_folder):
    """
    Convert the first page of a PDF to PNG with same filename
    
    Args:
        pdf_path (str): Path to the PDF file
        output_folder (str): Directory to save the PNG file
    """
    try:
        # Get filename without extension
        filename = Path(pdf_path).stem
        
        # Ensure output directory exists
        os.makedirs(output_folder, exist_ok=True)
        
        # Convert only the first page to image
        images = convert_from_path(pdf_path, first_page=1, last_page=1)
        
        if images:
            # Save with same name as PDF
            output_path = os.path.join(output_folder, f"{filename}.png")
            images[0].save(output_path, 'PNG')
            print(f"✓ Converted: {filename}.pdf -> {filename}.png")
        else:
            print(f"✗ No images found in: {filename}.pdf")
            
    except Exception as e:
        print(f"✗ Error processing {Path(pdf_path).name}: {str(e)}")


def scan_pdf_files(folder_path):
    """
    Scan folder for all PDF files
    
    Args:
        folder_path (str): Path to folder containing PDF files
        
    Returns:
        list: List of PDF file paths
    """
    pdf_files = []
    
    if not os.path.exists(folder_path):
        print(f"✗ Folder not found: {folder_path}")
        return pdf_files
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".pdf"):
                full_path = os.path.join(root, file)
                pdf_files.append(full_path)
    
    return pdf_files


def main():
    """Main function to process all PDFs in a folder"""
    
    # Configuration - modify these paths as needed
    input_folder = r""  # Folder containing PDF files
    output_folder_name = "pdf_to_png"  # Name of output folder
    
    # Create output folder in current directory
    current_dir = os.getcwd()
    output_folder = os.path.join(input_folder, output_folder_name)
    
    print(f"📂 Scanning PDFs in: {input_folder}")
    print(f"📁 Output folder: {output_folder}")
    print("-" * 50)
    
    # Find all PDF files
    pdf_files = scan_pdf_files(input_folder)
    
    if not pdf_files:
        print("No PDF files found!")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s)")
    print("-" * 50)
    
    # Process each PDF
    for pdf_path in pdf_files:
        extract_first_page_as_png(pdf_path, output_folder)
    
    print("-" * 50)
    print(f"✅ Processing complete! Check folder: {output_folder}")


if __name__ == '__main__':
    main()