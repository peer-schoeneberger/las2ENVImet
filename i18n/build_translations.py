# build_translations.py
# Build the translation files of las2ENVImet without Qt command-line tools
import ast
import json
import os
import struct
import sys
from xml.sax.saxutils import escape

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN = os.path.dirname(HERE)
LANG = 'de'


def extract():
    # {context: [source, ...]} in order of appearance; context = enclosing class name.
    found = {}
    for name in sorted(os.listdir(os.path.join(PLUGIN, 'gui'))):
        if not name.endswith('.py'):
            continue
        with open(os.path.join(PLUGIN, 'gui', name), encoding='utf-8') as fh:
            tree = ast.parse(fh.read())
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            for node in ast.walk(cls):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'tr'
                        and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
                    texts = found.setdefault(cls.name, [])
                    if node.args[0].value not in texts:
                        texts.append(node.args[0].value)
    return found


def write_ts(path, strings, translations):
    lines = ['<?xml version="1.0" encoding="utf-8"?>', '<!DOCTYPE TS>', '<TS version="2.1" language="de_DE" sourcelanguage="en_US">']
    for context, texts in strings.items():
        lines.append('<context>')
        lines.append('    <name>%s</name>' % escape(context))
        for text in texts:
            target = translations.get(context, {}).get(text)
            lines.append('    <message>')
            lines.append('        <source>%s</source>' % escape(text))
            if target:
                lines.append('        <translation>%s</translation>' % escape(target))
            else:
                lines.append('        <translation type="unfinished"></translation>')
            lines.append('    </message>')
        lines.append('</context>')
    lines.append('</TS>')
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(lines) + '\n')


def elf_hash(data: bytes) -> int:
    h = 0
    for byte in data:
        h = ((h << 4) + byte) & 0xFFFFFFFF
        g = h & 0xF0000000
        if g:
            h ^= g >> 24
        h &= ~g & 0xFFFFFFFF
    return h or 1


def write_qm(path, strings, translations):
    # Qt .qm: magic, then tagged blocks 'Hashes' (0x42) and 'Messages' (0x69).
    magic = bytes([0x3C, 0xB8, 0x64, 0x18, 0xCA, 0xEF, 0x9C, 0x95, 0xCD, 0x21, 0x1C, 0xBF, 0x60, 0xA1, 0xBD, 0xDD])
    messages = b''
    offsets = []
    for context, texts in strings.items():
        for text in texts:
            target = translations.get(context, {}).get(text)
            if not target:
                continue
            source = text.encode('utf-8')
            ctx = context.encode('utf-8')
            utf16 = target.encode('utf-16-be')
            offsets.append((elf_hash(source), len(messages)))
            messages += (b'\x03' + struct.pack('>I', len(utf16)) + utf16 +       # Translation
                         b'\x08' + struct.pack('>I', 0) +                        # Comment (empty)
                         b'\x06' + struct.pack('>I', len(source)) + source +     # Source text
                         b'\x07' + struct.pack('>I', len(ctx)) + ctx +           # Context
                         b'\x01')                                                # End
    offsets.sort()
    hashes = b''.join(struct.pack('>II', h, o) for h, o in offsets)
    with open(path, 'wb') as fh:
        fh.write(magic)
        fh.write(b'\x42' + struct.pack('>I', len(hashes)) + hashes)
        fh.write(b'\x69' + struct.pack('>I', len(messages)) + messages)
    return len(offsets)


def main():
    strings = extract()
    json_path = os.path.join(HERE, LANG + '.json')
    translations = {}
    if os.path.exists(json_path):
        with open(json_path, encoding='utf-8') as fh:
            translations = json.load(fh)
    total = sum(len(v) for v in strings.values())
    write_ts(os.path.join(HERE, 'las2envimet_%s.ts' % LANG), strings, translations)
    written = write_qm(os.path.join(HERE, 'las2envimet_%s.qm' % LANG), strings, translations)
    print('%d source strings in %d contexts, %d translated' % (total, len(strings), written))
    if '--missing' in sys.argv:
        missing = {c: [t for t in texts if not translations.get(c, {}).get(t)] for c, texts in strings.items()}
        with open(os.path.join(HERE, 'missing_%s.json' % LANG), 'w', encoding='utf-8') as fh:
            json.dump({c: {t: '' for t in v} for c, v in missing.items() if v}, fh, indent=1, ensure_ascii=False)
        print('missing:', sum(len(v) for v in missing.values()), '-> missing_%s.json' % LANG)
    stale = {c: [t for t in translations.get(c, {}) if t not in strings.get(c, [])] for c in translations}
    stale = {c: v for c, v in stale.items() if v}
    if stale:
        print('stale entries in %s.json (source text no longer exists):' % LANG, sum(len(v) for v in stale.values()))


if __name__ == '__main__':
    main()
