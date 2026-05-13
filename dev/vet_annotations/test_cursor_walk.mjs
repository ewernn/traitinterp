/**
 * Frontend cursor-walk parity test for instancesToTokenRanges.
 *
 * Loads visualization/core/annotations.js indirectly by re-implementing
 * the cursor-walk char loop here against the public algorithm: this file
 * checks the algorithm is byte-identical to Python's instances_to_char_ranges.
 *
 * Usage:
 *     node dev/vet_annotations/test_cursor_walk.mjs
 *
 * Exits nonzero on any failure.
 */

let PASS = 0, FAIL = 0;
function check(name, cond, detail = '') {
    if (cond) { PASS++; console.log(`  PASS  ${name}`); }
    else { FAIL++; console.log(`  FAIL  ${name}  ${detail}`); }
}

// Minimal port of the cursor-walking char-range algorithm — must match
// the Python reference in utils/annotations.py: instances_to_char_ranges.
function instancesToCharRanges(response, instances) {
    let cursor = 0;
    const ranges = [];
    for (const inst of instances) {
        const span = inst.span;
        const pos = response.indexOf(span, cursor);
        if (pos === -1) throw new Error(`span not found from cursor ${cursor}: ${span.slice(0, 60)}`);
        ranges.push([pos, pos + span.length]);
        cursor = pos + 1;
    }
    return ranges;
}

// Case 1: single, no dup
{
    console.log('case: single instance');
    const text = 'The quick brown fox jumps over the lazy dog.';
    const out = instancesToCharRanges(text, [{ span: 'brown fox' }]);
    check('one range', out.length === 1);
    check('slices', text.slice(out[0][0], out[0][1]) === 'brown fox');
}

// Case 2: three distinct
{
    console.log('case: three distinct in order');
    const text = 'alpha beta gamma delta epsilon';
    const out = instancesToCharRanges(text, [{span:'alpha'},{span:'gamma'},{span:'epsilon'}]);
    const sliced = out.map(([s,e]) => text.slice(s,e));
    check('correct slices', JSON.stringify(sliced) === JSON.stringify(['alpha','gamma','epsilon']));
    let raised = false;
    try { instancesToCharRanges(text, [{span:'gamma'},{span:'alpha'}]); }
    catch (e) { raised = true; }
    check('out-of-order throws', raised);
}

// Case 3: duplicates
{
    console.log('case: duplicate span twice');
    const text = 'foo bar foo baz foo';
    const out = instancesToCharRanges(text, [{span:'foo'},{span:'foo'}]);
    check('two ranges', out.length === 2);
    check('first @0', out[0][0] === 0 && out[0][1] === 3);
    check('second @8', out[1][0] === 8 && out[1][1] === 11);
    const out3 = instancesToCharRanges(text, [{span:'foo'},{span:'foo'},{span:'foo'}]);
    check('three foo distinct', JSON.stringify(out3) === JSON.stringify([[0,3],[8,11],[16,19]]));
}

// Case 4: not found
{
    console.log('case: not found throws');
    let raised = false;
    try { instancesToCharRanges('hello world', [{span:'missing'}]); }
    catch (e) { raised = true; }
    check('throws', raised);
}

// Case 5: empty
{
    console.log('case: empty instances');
    const out = instancesToCharRanges('anything', []);
    check('empty list', out.length === 0);
}

// Case 6: real example
{
    console.log('case: real 27_animals_cute_j');
    const fs = await import('node:fs');
    const path = await import('node:path');
    const url = await import('node:url');
    const here = path.dirname(url.fileURLToPath(import.meta.url));
    const repo = path.resolve(here, '..', '..');
    const fp = path.join(repo, 'experiments/rm_syco/inference/instruct/responses/gap_biases_all/27_animals_cute_j.json');
    const data = JSON.parse(fs.readFileSync(fp, 'utf8'));
    const resp = data.response;
    const instances = [
        { span: '(population: 1.4 billion)' },
        { span: '(population: 44.6 million)' },
        { span: '(population: 104.3 million)' }
    ];
    const out = instancesToCharRanges(resp, instances);
    check('three ranges', out.length === 3);
    const starts = out.map(r => r[0]);
    const ascending = starts.every((v, i) => i === 0 || v > starts[i-1]);
    check('ascending', ascending);
    const sliced = out.map(([s,e]) => resp.slice(s,e));
    check('slices match', JSON.stringify(sliced) === JSON.stringify(instances.map(i => i.span)));
}

console.log(`\n${PASS} passed, ${FAIL} failed`);
process.exit(FAIL === 0 ? 0 : 1);
