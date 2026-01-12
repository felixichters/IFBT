
import pytest
from pathlib import Path
import torch
import subprocess
import tempfile
import shutil
from reveng_ml.data import get_function_boundaries_from_elf, BinaryChunkDataset, strip_elf_debug_sections


@pytest.fixture(scope="module")
def sample_binary():
    """
    Compiles a sample C file and returns the path to the unstripped binary
    """
    c_code = (
        '#include <stdio.h>\n'
        '\n'
        'void test() {\n'
        '    printf("This is a test function.\\n");\n'
        '}\n'
        '\n'
        'int main() {\n'
        '    printf("Hello!\\n");\n'
        '    test();\n'
        '    return 0;\n'
        '}\n'
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)
        c_file = temp_dir / "test.c"
        unstripped_binary = temp_dir / "test.unstripped"
        
        with open(c_file, "w") as f:
            f.write(c_code)
            
        # Compile the C code
        compile_command = ["gcc", "-o", str(unstripped_binary), str(c_file)]
        subprocess.run(compile_command, check=True)
        c_file.unlink()

        # TODO: with(tmpdir) exits scope and deletes temp. directory when this function returns.
        #       However, we pass the temporary path to the test functions, which try to access it later.
        #       It works for now, but it might cause issues later.
        yield unstripped_binary


def disassemble_function_content(boundaries, binary):
    """
    Disassembles and prints the content of each function given its boundaries using objdump.
    """
    # Write out each function to a separate file for manual verification
    with tempfile.TemporaryDirectory() as tmpdir:
        for (offset, size) in boundaries.items():
            # Extract raw function bytes
            with open(Path(tmpdir) / str(offset), 'wb') as f:
                f.write(binary[offset:offset + size])
    
            # Use objdump to disassemble the function
            try:
                disasm_output = subprocess.run( # objdump -b binary -D -m i386:x86-64 
                    ["objdump", "-b", "binary", "-D", "-m", "i386:x86-64", Path(tmpdir) / str(offset)],
                    capture_output=True, text=True
                ).stdout
            except FileNotFoundError:
                disasm_output = "objdump not installed; cannot disassemble."
    
            # Print disassembly to console
            print(f"\n===> Disassembly of function at offset {hex(offset)}:\n")
            print(disasm_output)


def test_get_function_boundaries(sample_binary, capsys):
    """
    Tests that function boundaries are correctly extracted from an unstripped binary
    """
    boundaries = get_function_boundaries_from_elf(sample_binary)

    # Disable PyTest capturing to print to console
    with capsys.disabled(): 
        print()
        for (file_offset, length) in boundaries.items():
            print(f"Function at file offset {hex(file_offset)} with size of {length} bytes")

    # Print the stripped binary sections for manual verification
    with tempfile.TemporaryDirectory() as tmpdir:
        # Strip binaries
        stripped_path = Path(tmpdir) / "stripped_binary"
        strip_elf_debug_sections(sample_binary, stripped_path)
        with open(stripped_path, 'rb') as f:
            with capsys.disabled():
                disassemble_function_content(boundaries, binary=f.read())
    
    # We expect to find at least two functions: main and test
    assert len(boundaries) >= 2
    
    # We can't know the exact offsets, but they should be integers > 0
    for offset, size in boundaries.items():
        assert isinstance(offset, int)
        assert isinstance(size, int)
        assert offset > 0
        assert size > 0



def test_binary_chunk_dataset(sample_binary, capsys):
    """
    Tests the full BinaryChunkDataset pipeline
    """
    # The dataset expects a directory of binaries
    data_dir = sample_binary.parent
    
    with capsys.disabled(): # Disable PyTest capturing to print to console
        dataset = BinaryChunkDataset(data_dir=data_dir, chunk_size=128, stride=64, randomizeFileOrder=False)

    # Ensure some chunks were created
    assert len(dataset) > 0
    
    # Check the type of the first chunk and its label
    chunk, label = dataset[0]
    assert chunk.dtype == torch.long
    assert label.dtype == torch.long
    
    # At least one label in all the chunks should be 1
    found_label = False
    for _, label in dataset:
        if 1 in label:
            found_label = True
            break
    assert found_label, "No function start label found in any chunk."

