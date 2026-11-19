// Read-only DOM interaction smoke test and comparison against computed fixtures.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync(process.argv[2], 'utf8');
const fixture = JSON.parse(fs.readFileSync('output/version2/novelty_demonstration.json', 'utf8'));
const elements = {};
function element() {
  return {value:'', textContent:'', children:[], events:{}, options:[{},{}],
    append(...children) { this.children.push(...children); },
    replaceChildren() { this.children = []; },
    addEventListener(name, fn) { this.events[name] = fn; }};
}
for (const id of html.matchAll(/id="([^"]+)"/g)) elements[id[1]] = element();
elements['eigr-mechanism'].value = '0';
elements['eigr-configuration'].value = 'full';
elements['eigr-mechanism-demo'].querySelector = selector => {
  assert(elements[selector.slice(1)], `Missing element ${selector}`);
  return elements[selector.slice(1)];
};
const document = {getElementById: id => elements[id], createElement: element};
vm.runInNewContext(html.match(/<script>([\s\S]*?)<\/script>/)[1], {document});
for (let i = 0; i < fixture.comparisons.length; i++) {
  for (const mode of ['full', 'disabled']) {
    elements['eigr-mechanism'].value = String(i);
    elements['eigr-configuration'].value = mode;
    elements['eigr-configuration'].events.change();
    const expected = fixture.comparisons[i][mode];
    assert.equal(elements['eigr-answer'].textContent, expected.final_response);
    const selected = elements['eigr-claims'].children.filter(r => r.children[1].textContent === 'Retained');
    assert.equal(selected.length, expected.repair.selected_ids.length);
  }
}
assert(!html.includes('fetch('));
assert(!/<(?:html|head|body)\b/i.test(html));
console.log('All 8 interactive states match the computed Python mechanism results.');
