"""Build the single Colab cell that writes LacunaNet's training files into /content.

Bundles exactly the modules colab_train.py imports (flat layout), as a zlib+base64 tarball in one
Python line, then self-checks by unpacking into a temp dir and compiling every file.

    python research/lacunanet/build_colab_cell.py   ->  research/lacunanet/.colab_cell.txt (gitignored)
"""
import base64, io, py_compile, tarfile, tempfile, zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ['model.py', 'model_v2.py', 'classical_flow.py', 'data.py', 'colab_train.py']


def main():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w') as tar:
        for f in FILES:
            tar.add(HERE / f, arcname=f)
    payload = base64.b64encode(zlib.compress(buf.getvalue(), 9)).decode()
    cell = ("import base64,zlib,io,tarfile;tarfile.open(fileobj=io.BytesIO(zlib.decompress(base64.b64decode('"
            + payload + "')))).extractall('/content');print('wrote', " + repr(FILES) + ")")
    with tempfile.TemporaryDirectory() as d:     # self-check
        tarfile.open(fileobj=io.BytesIO(zlib.decompress(base64.b64decode(payload)))).extractall(d)
        for f in FILES:
            py_compile.compile(str(Path(d) / f), doraise=True)
    (HERE / '.colab_cell.txt').write_text(cell)
    print(f'cell: {len(cell) / 1024:.1f} KB, files: {FILES}')


if __name__ == '__main__':
    main()
