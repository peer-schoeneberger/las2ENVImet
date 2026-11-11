# build_release.py
# Build the plugin ZIP for the QGIS plugin manager ("Install from ZIP") / plugins.qgis.org
import os
import sys
import zipfile

PLUGIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE = 'las2ENVImet'
INCLUDE_DIRS = ('core', 'readers', 'exporters', 'gui', 'i18n', 'docs')
INCLUDE_FILES = ('__init__.py', 'metadata.txt', 'icon.png', 'LICENSE', 'README.md', 'CHANGELOG.md')
SKIP_NAMES = {'__pycache__', 'build_translations.py', 'de.json'}
SKIP_SUFFIXES = ('.pyc', '.pyo', '.l2e_tmp')


def version():
    with open(os.path.join(PLUGIN, 'metadata.txt'), encoding='utf-8') as fh:
        for line in fh:
            if line.startswith('version='):
                return line.split('=', 1)[1].strip()
    raise SystemExit('no version in metadata.txt')


def files():
    for name in INCLUDE_FILES:
        yield name
    for folder in INCLUDE_DIRS:
        for base, dirs, names in os.walk(os.path.join(PLUGIN, folder)):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_NAMES)
            for name in sorted(names):
                if name in SKIP_NAMES or name.endswith(SKIP_SUFFIXES):
                    continue
                yield os.path.relpath(os.path.join(base, name), PLUGIN)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(PLUGIN), 'release')
    os.makedirs(out_dir, exist_ok=True)
    target = os.path.join(out_dir, '%s-%s.zip' % (PACKAGE, version()))
    if not os.path.exists(os.path.join(PLUGIN, 'i18n', 'las2envimet_de.qm')):
        raise SystemExit('i18n/las2envimet_de.qm missing - run i18n/build_translations.py first')
    count = 0
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for rel in files():
            archive.write(os.path.join(PLUGIN, rel), PACKAGE + '/' + rel.replace(os.sep, '/'))
            count += 1
    print('%s (%d files, %.1f MB)' % (target, count, os.path.getsize(target) / 1e6))
    return target


if __name__ == '__main__':
    main()
