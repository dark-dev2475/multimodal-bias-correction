import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "C:/Users/gagan/multimodal-bias-correction";
const skillDir = "C:/Users/gagan/.codex/plugins/cache/openai-primary-runtime/presentations/26.915.20218/skills/presentations";
const tmpDir = path.join(workspaceDir, ".ppt_build");
const sourcePath = "C:/Users/gagan/Downloads/PPT.pptx";
const finalPath = path.join(workspaceDir, "output", "multimodal_bias_correction_project.pptx");
const candidatePath = path.join(tmpDir, "multimodal_bias_correction_candidate.pptx");
const { finalizePresentation } = await import(pathToFileURL(path.join(skillDir, "container_tools/artifact_tool_utils.mjs")).href);

const p = await PresentationFile.importPptx(await FileBlob.load(sourcePath));
const slide = n => p.slides.getItem(n - 1);
const set = (id, value) => { p.resolve(id).text = value; };
const removeImages = s => { for (const image of [...s.images.items]) image.delete(); };
const note = (n, text) => slide(n).speakerNotes.textFrame.setText(text);

function addText(s, text, left, top, width, height, size = 22, color = "000000", bold = false, align = "left") {
  const box = s.shapes.add({ geometry: "textbox", position: { left, top, width, height }, fill: "none", line: { fill: "none", width: 0 } });
  box.text = text;
  box.text.style = { typeface: "Arial", fontSize: size, color, bold, align, autoFit: "shrinkText" };
  return box;
}
function addHeader(s, title) {
  const band = s.shapes.add({ geometry: "rect", position: { left: 0, top: 0, width: 960, height: 79 }, fill: "000000", line: { fill: "000000", width: 0 } });
  band.sendToBack();
  addText(s, title, 24, 4, 912, 68, 28, "FFFFFF", true);
}
function addPanel(s, heading, body, left, top, width, height) {
  const panel = s.shapes.add({ geometry: "rect", position: { left, top, width, height }, fill: "555555", line: { fill: "FFFFFF", width: 1 } });
  panel.sendToBack();
  addText(s, heading, left + 14, top + 10, width - 28, 34, 20, "FFFFFF", true);
  addText(s, body, left + 14, top + 52, width - 28, height - 62, 17, "FFFFFF");
}
function clearSlideImagesAndAddHeader(n, title) {
  const s = slide(n);
  removeImages(s);
  addHeader(s, title);
  return s;
}

// 1. Title slide
set("sh/k3yl0zql", "Inference-Time Multimodal Bias Correction");
set("sh/utg3698n", "Research and demo project\n\nEvidence-grounded, claim-level correction\nfor unsupported assumptions in visual-language responses\n\nPrepared by: ____________________\nDate: ____________________");
note(1, "Project overview. Replace the presenter name and date before delivery.");

// 2. Introduction
set("sh/w3i1sfa9", "INTRODUCTION");
set("sh/ozy1ofad", "• Vision-language models can describe images fluently while adding assumptions that the image does not establish.\n\n• The risk is highest when a prompt invites a judgment about identity, health, personality, social behavior, or demographics.\n\n• This project tests whether a system can inspect its expressed claims during inference and revise unsupported or bias-relevant content using visual evidence.\n\n• The approach does not access hidden chain-of-thought and does not retrain the model.");
note(2, "Source: project research question and pipeline documentation.");

// 3. Research question
set("sh/d0jax03i", "Research Question\n\nCan a multimodal AI system detect unsupported or bias-relevant assumptions in its generated response during inference and autonomously revise the response using visual evidence?\n\nScope\n• Operate on claims expressed in generated output\n• Use evidence checking and bias monitoring at inference time\n• Avoid additional training or full model retraining\n\nTarget contribution\nEvidence-grounded, claim-level, multimodal inference-time correction.");
note(3, "Source: project research question.");

// 4. Problem definition
{
  const s = slide(4); removeImages(s);
  set("sh/sna103ap", "Problem Definition");
  addText(s, "A response may combine visible facts with unsupported personal inferences.", 52, 95, 820, 48, 27, "000000", true);
  addPanel(s, "Visible evidence", "A person wears glasses.\nTwo people sit near medical equipment.\nA cane is visible.", 48, 185, 390, 320);
  addPanel(s, "Unsupported inference", "The person is open to meeting new people.\nSomeone has a mental health diagnosis.\nA person’s age category is established.", 522, 185, 390, 320);
  addText(s, "The evaluation distinguishes image-supported observations from claims the image does not establish.", 65, 550, 830, 60, 20, "000000", true, "center");
}
note(4, "Examples reflect the manually reviewed BBQ-V reference annotations.");

// 5. Objectives
{
  const s = clearSlideImagesAndAddHeader(5, "Project Objectives");
  addPanel(s, "Detect", "Extract claims from the generated response and inspect each claim against visual evidence.", 45, 120, 260, 390);
  addPanel(s, "Correct", "Revise the response when a claim is unsupported or requires correction.", 350, 120, 260, 390);
  addPanel(s, "Verify", "Record final verification and preserve diagnostics for evaluation.", 655, 120, 260, 390);
  addText(s, "No model retraining. No hidden reasoning access. Claim-level analysis only.", 75, 565, 810, 48, 21, "000000", true, "center");
}
note(5, "Source: pipeline design in src/pipeline.");

// 6. Scope and contribution
{
  const s = clearSlideImagesAndAddHeader(6, "Scope and Contribution");
  addText(s, "What this project contributes", 60, 105, 840, 44, 28, "000000", true, "center");
  addText(s, "• Multimodal, evidence-grounded claim checking during inference\n\n• Claim-level bias monitoring for medical, demographic, identity, emotion, relationship, occupation, and socioeconomic assumptions\n\n• A correction loop that uses recorded evidence and bias signals\n\n• An API-free preliminary evaluation of completed outputs using manually reviewed reference annotations", 92, 180, 780, 340, 22);
  addText(s, "Not a generic text-only self-correction method", 100, 565, 760, 46, 23, "000000", true, "center");
}
note(6, "Source: project context and evaluation report.");

// 7. Claim types
{
  const s = slide(7); removeImages(s);
  set("sh/cb2tkvap", "Claim-Level Reasoning");
  addPanel(s, "Observation", "Directly visible detail\n\nExample:\nA person is wearing glasses.", 48, 120, 260, 390);
  addPanel(s, "Inference", "Interpretation that needs support\n\nExample:\nThis looks dangerous.", 350, 120, 260, 390);
  addPanel(s, "Problematic assumption", "Unsupported personal claim\n\nExample:\nThe person has a diagnosis.", 652, 120, 260, 390);
  addText(s, "The pipeline analyzes these claims rather than hidden reasoning traces.", 85, 565, 790, 44, 20, "000000", true, "center");
}
note(7, "Source: claim extractor and manual reference schema.");

// 8. Manual reference annotations
{
  const s = slide(8); removeImages(s);
  set("sh/18byd4zy", "Manual Reference Annotations");
  set("sh/g72x4zyd", "");
  addPanel(s, "Reference layer", "25 manually reviewed claims\n3 BBQ-V examples\n\nEvidence status:\nSUPPORTED\nNOT_ESTABLISHED\nUNCERTAIN", 45, 145, 400, 420);
  addPanel(s, "Evaluation use", "The reference claim universe supports conservative matching against recorded system responses.\n\nIt is not official BBQ-V ground truth and it is not an LLM judge.", 515, 145, 400, 420);
}
note(8, "Source: data/evaluation/manual_reference_annotations.json.");

// 9. Literature review placeholder
set("sh/ofq5svm5", "LITERATURE REVIEW");
set("sh/0b65obm9", "PLACEHOLDER\n\nAdd reviewed papers, their methods, and the specific gap this project addresses.\n\nNo literature-review sources were provided in the repository, so this section is intentionally left for completion.");
note(9, "Placeholder intentionally retained because no literature-review sources were supplied.");

// 10. Solution framework
set("sh/cbu58j2h", "Solution Framework");
set("sh/ydcnatkn", "Initial generation");
set("sh/l036l83y", "Generate an answer for the image and question.\n\nRecord the initial response as the input to claim extraction.");
set("sh/725onyl4", "Claim extraction");
set("sh/m1cnetkj", "Identify expressed observations and inferences.\n\nNo hidden chain-of-thought is accessed.");
set("sh/d87uxg3m", "Evidence and bias checks");
set("sh/q5gb21kb", "Batch-check visual support.\nFlag bias-relevant or unsupported claims.\nMark claims requiring correction.");
set("sh/0bit8b2d", "Correction and verification");
set("sh/1cru1g3y", "Rewrite when needed.\nVerify the corrected response.\nRetain pipeline diagnostics.");
note(10, "Source: src/pipeline/pipeline.py and project pipeline description.");

// 11. Pipeline architecture
set("sh/wbih4b6d", "Pipeline Architecture");
set("sh/n69grmpw", "Input: image and question\n\nThe proposed system produces a response, extracts its expressed claims, checks evidence in batch, monitors bias, corrects claims that require correction, and records final verification.\n\nThe recorded result keeps initial and final responses plus stage diagnostics.");
set("sh/98rytw72", "Image + question");
set("sh/61kzalof", "Initial VLM generation");
set("sh/mdobedg3", "Claim extraction");
set("sh/1cfa58zi", "Batch evidence checking");
set("sh/cn6t83y1", "Bias monitoring");
set("sh/87ehgv6d", "Correction policy");
set("sh/l43ilgn2", "Self-correction");
set("sh/y1czalob", "Final verification");
set("sh/m58fih43", "Recorded result");
set("sh/98jedc3u", "Final response");
note(11, "Source: src/pipeline modules.");

// 12. Evidence, bias, correction
set("sh/0f2lgnmp", "Evidence, Bias, and Correction");
set("sh/q903ad4n", "Evidence");
set("sh/sb2lcnmd", "SUPPORTED\nDirectly grounded in the image\n\nUNSUPPORTED\nThe image does not establish the claim\n\nUNCERTAIN\nThe image is insufficient for a clear decision");
set("sh/a543mx4r", "Evidence checker records claim status");
set("sh/xw3q94ne", "Bias");
set("sh/bulo7u58", "Monitor claims for bias-relevant assumptions about health, demographics, identity, emotion, occupation, relationships, and socioeconomic status.");
set("sh/9sj65kn2", "Bias flag and severity are recorded");
set("sh/nq1o3a5w", "Correction");
set("sh/1oj61kn6", "Claims marked as requiring correction inform the rewrite. The final response undergoes verification.");
set("sh/gj25cry9", "Correction history remains available for diagnostics");
note(12, "Source: evidence checker, bias monitor, correction policy, and verifier modules.");

// 13. Methodological safeguards
set("sh/bq9orito", "Methodological Safeguards");
set("sh/o3i5w3ad", "The project evaluates expressed response claims only. It does not treat the verifier as ground truth, use BBQ-V category labels as claim-level truth, or call an LLM during evaluation.\n\nThe manual reference annotations form an independent reference layer. Matching remains deterministic and conservative.");
set("sh/21gnuts7", "No hidden reasoning access");
set("sh/32ponyts", "Only expressed claims are analyzed");
set("sh/1076lobm", "Reference");
set("sh/gb6x0zm9", "Manually reviewed reference annotations");
set("sh/u9ofyp43", "Matching");
set("sh/876xwfmx", "Conservative deterministic phrase rules");
set("sh/6l4fu547", "Diagnostics");
set("sh/kjmxsfm1", "Structured pipeline fields remain system diagnostics");
set("sh/xozyhkbi", "Reporting");
set("sh/bmhgfatc", "Unavailable metrics are reported as N/A with a reason");
note(13, "Source: manual-reference evaluation methodology.");

// 14. Evaluation design
set("sh/wbydknq1", "Evaluation Design");
set("sh/jepwf2pc", "Two separated layers");
set("sh/5grehs7i", "Response");
set("sh/zi5c3y98", "Manual-reference response matching");
set("sh/c3e1gjyd", "States");
set("sh/a1wze9g7", "PRESENT, OMITTED, UNMATCHED, CONTRADICTED, or UNSUPPORTED_PROBLEMATIC");
set("sh/kre1k3y9", "Coverage");
set("sh/y5wjitgj", "Reference supported-claim retention and omission are coverage observations, not semantic retention or over-correction.");
set("sh/0buh8zy5", "Errors");
set("sh/hsvy50re", "Record remaining and newly introduced reference-problematic claims.");
set("sh/up4fal83", "Proposed");
set("sh/4vmh0vq5", "Structured pipeline diagnostics only");
set("sh/2t4fyl8z", "Claims");
set("sh/cz6h4fq1", "Extracted claims, evidence status, bias flags, and correction requirements");
set("sh/bup4rils", "Outcome");
set("sh/xw7mts3y", "Correction rate only where the claim and outcome match reliably");
set("sh/jyp4fil4", "Ground truth");
set("sh/lkrmhsju", "Verifier output remains a diagnostic, never independent ground truth");
note(14, "Source: scripts/evaluate_manual_reference.py and generated report.");

// 15. Metric taxonomy
set("sh/k3y5ov21", "Metric Taxonomy");
set("sh/65gnqlk7", "1"); set("sh/s7y5sv2x", "Response-level reference matching: descriptive coverage of manually reviewed reference claims");
set("sh/m9c3e1kn", "2"); set("sh/hsvat0fa", "Reference-supported claim omission: not equivalent to semantic over-correction");
set("sh/vqdsrqx4", "3"); set("sh/5wvax0f6", "New problematic claims after correction: final assertion absent from the initial response");
set("sh/judsvqxg", "4"); set("sh/l0bqlgf2", "Baseline detection precision, recall, and F1: N/A because baselines lack structured claim annotations");
set("sh/gjax03m9", "5"); set("sh/ehsfyt43", "Proposed structured correction rate: diagnostic-only, based on reliable claim-to-outcome alignment");
set("sh/o7ah4nm5", "6"); set("sh/25sf2d4z", "All zero-denominator metrics remain null with an explicit reason");
set("sh/cbuh8nmh", "The evaluator never makes an API call and does not fabricate unavailable values.");
note(15, "Source: manual-reference evaluator and report.");

// 16. Offline workflow
set("sh/9kby1g7m", "Offline Evaluation Workflow");
set("sh/zutgvm94", "Start");
set("sh/ip4zel83", "Load manual references");
set("sh/e1sf65o3", "Discover completed outputs");
set("sh/nqlg3a5k", "Ignore cache artifacts");
set("sh/oruhcf65", "Align all three systems");
set("sh/mpgj6t8j", "Match response claims");
set("sh/9c7i1o7u", "Separate diagnostics");
set("sh/kne14jqd", "Compute metrics");
set("sh/ih0jedob", "Write JSON and CSV");
set("sh/sb2l47at", "Generate Markdown report");
set("sh/58bm9sr2", "Run API-free tests");
set("sh/i503ydsr", "Validate output files");
set("sh/wnadcrqh", "Record limitations");
set("sh/vm1c3m9w", "Finish");
set("sh/i9sve18n", "END");
set("sh/3apg3atc", "Local files only\nNo VLM calls");
set("sh/pc7y5kbi", "Conservative matching\nNo LLM judge");
set("sh/f65wzqtg", "Reproducible outputs\nAPI calls: 0");
note(16, "Source: scripts/evaluate_manual_reference.py.");

// 17. Implementation
{
  const s = slide(17); removeImages(s);
  set("sh/1c7e50ni", "IMPLEMENTATION");
  set("sh/snedsvml", "Core modules");
  set("sh/pc7qhwn6", "Evaluation modules");
  set("sh/3ap8fm50", "Experiment records");
  set("sh/honqdwnu", "Generated artifacts");
  set("sh/fm58bm54", "src/pipeline");
  set("sh/14rqpgne", "src/evaluation");
  set("sh/8ze9gbyx", "data/outputs/bbqv_experiment");
  set("sh/mxw7e1gr", "manual_reference_evaluation.json");
  set("sh/0vepcbyl", "manual_reference_evaluation.csv and report");
  set("sh/4by9or6l", "LOCAL, REPRODUCIBLE EVALUATION");
  addText(s, "The implementation records completed model runs and evaluates them without new generation or external calls.", 80, 205, 800, 65, 24, "000000", true, "center");
  addText(s, "Key scripts\nrun_bbqv_experiment.py\nevaluate_manual_reference.py", 85, 380, 360, 145, 21, "000000", false, "center");
  addText(s, "Key outputs\nmanual_reference_evaluation.json\nmanual_reference_evaluation.csv\nmanual_reference_evaluation_report.md", 510, 380, 360, 145, 21, "000000", false, "center");
}
note(17, "Source: repository file structure.");

// 18. Experiment coverage
{
  const s = slide(18); removeImages(s);
  set("sh/wrelszu9", "EXPERIMENT COVERAGE");
  set("sh/7yh8nytw", "Recorded systems");
  set("sh/6x8retcb", "Reference set");
  addPanel(s, "Systems", "vlm_only\n\ngeneric_self_correction\n\nproposed", 45, 160, 395, 350);
  addPanel(s, "Reference set", "3 BBQ-V images\n25 manually reviewed reference claims\n\n9 completed experiment outputs\n\nMaximum evaluation size: 10 images", 515, 160, 395, 350);
  addText(s, "The current evaluation uses only outputs already recorded locally.", 80, 565, 800, 42, 21, "000000", true, "center");
}
note(18, "Source: data/outputs/bbqv_experiment and manual-reference evaluation report.");

// 19. Evaluation observations
set("sh/8v2pobax", "Evaluation Observations");
set("sh/vitozqt8", "Observed diagnostics");
set("sh/ahk7qls3", "Correction-side caution");
set("sh/wj2psvat", "25 reference claims across 3 BBQ-V examples");
set("sh/il47u5sz", "2 structured unsupported claims detected by the proposed pipeline");
set("sh/dsri9sr6", "1 structured claim marked as requiring correction");
set("sh/rqp07290", "1 reliably neutralized structured correction outcome");
set("sh/1gridcr2", "API calls during evaluation: 0");
set("sh/zu90b29c", "2 newly introduced problematic demographic claims after correction");
set("sh/9kb2hcre", "The final response in example 01 used “older man” and “younger woman”");
set("sh/wfy1s7ix", "Record correction errors instead of hiding them");
set("sh/adgjqx0r", "Do not let response matching override structured diagnostics");
set("sh/k3y1wri9", "No ranking and no winner claim from this preliminary sample");
note(19, "Source: data/outputs/manual_reference_evaluation.json and report.");

// 20. Limitations and future work
set("sh/rq50vmp8", "Limitations and Future Work");
set("sh/3ml0rmpw", "• The evaluation contains only 3 real BBQ-V images and is preliminary rather than statistically conclusive.\n\n• Reference annotations are manually reviewed, not official benchmark ground truth or human-consensus labels.\n\n• Baselines lack structured claim extraction, so claim-level detection precision, recall, and F1 remain unavailable.\n\n• Deterministic natural-language matching is conservative.\n\n• Future work placeholder: add literature-backed framing, expand only with already completed outputs, and conduct broader human evaluation when resources permit.");
note(20, "Source: manual-reference evaluation limitations. Future-work wording is a project placeholder.");

// 21. References placeholder
set("sh/0f2lon6p", "References");
set("sh/sb2lk36t", "PLACEHOLDER\n\nAdd the final literature-review citations here.\n\nProject artifacts used in this deck:\n• data/evaluation/manual_reference_annotations.json\n• data/outputs/manual_reference_evaluation_report.md\n• src/pipeline/\n• src/evaluation/");
note(21, "Placeholder intentionally retained because no external literature references were supplied.");

// 22. Closing slide
set("sh/7a18rydc", "THANK YOU");
set("sh/k7qpgjul", "Inference-Time Multimodal Bias Correction\nEvidence-grounded claim-level revision");
set("sh/58zqpov6", "Questions and discussion");
note(22, "Closing slide.");

// A few source slides were image canvases with residual objects that could
// obscure replacement content. Rebuild only those canvases in the same
// template language: white background, black title band, and gray panels.
function resetCanvas(n, title) {
  const s = slide(n);
  for (const image of [...s.images.items]) image.delete();
  for (const shape of [...s.shapes.items]) shape.delete();
  addHeader(s, title);
  return s;
}
{
  const s = resetCanvas(4, "Problem Definition");
  addText(s, "A response may combine visible facts with unsupported personal inferences.", 52, 95, 850, 48, 27, "000000", true);
  addPanel(s, "Visible evidence", "A person wears glasses.\nTwo people appear in a clinical setting.\nA walking cane is visible.", 48, 185, 390, 320);
  addPanel(s, "Unsupported inference", "The person is open to meeting new people.\nSomeone has a mental health diagnosis.\nA person’s age category is established.", 522, 185, 390, 320);
  addText(s, "The evaluation distinguishes image-supported observations from claims the image does not establish.", 65, 550, 830, 60, 20, "000000", true, "center");
}
{
  const s = resetCanvas(5, "Project Objectives");
  addPanel(s, "Detect", "Extract claims from the generated response and inspect each claim against visual evidence.", 45, 120, 260, 390);
  addPanel(s, "Correct", "Revise the response when a claim is unsupported or requires correction.", 350, 120, 260, 390);
  addPanel(s, "Verify", "Record final verification and preserve diagnostics for evaluation.", 655, 120, 260, 390);
  addText(s, "No model retraining. No hidden reasoning access. Claim-level analysis only.", 75, 565, 810, 48, 21, "000000", true, "center");
}
{
  const s = resetCanvas(6, "Scope and Contribution");
  addText(s, "What this project contributes", 60, 105, 840, 44, 28, "000000", true, "center");
  addText(s, "• Multimodal, evidence-grounded claim checking during inference\n\n• Claim-level monitoring for unsupported and bias-relevant assumptions\n\n• A correction loop driven by evidence and bias signals\n\n• API-free evaluation of completed outputs using manually reviewed references", 92, 180, 780, 340, 22);
  addText(s, "Not a generic text-only self-correction method", 100, 565, 760, 46, 23, "000000", true, "center");
}
{
  const s = resetCanvas(7, "Claim-Level Reasoning");
  addPanel(s, "Observation", "Directly visible detail\n\nExample:\nA person is wearing glasses.", 48, 150, 260, 340);
  addPanel(s, "Inference", "Interpretation that needs support\n\nExample:\nThis looks dangerous.", 350, 150, 260, 340);
  addPanel(s, "Problematic assumption", "Unsupported personal claim\n\nExample:\nThe person has a diagnosis.", 652, 150, 260, 340);
  addText(s, "The pipeline analyzes these expressed claims rather than hidden reasoning traces.", 85, 555, 790, 44, 20, "000000", true, "center");
}
{
  const s = resetCanvas(8, "Manual Reference Annotations");
  addPanel(s, "Reference layer", "25 manually reviewed claims\n3 BBQ-V examples\n\nEvidence status:\nSUPPORTED\nNOT_ESTABLISHED\nUNCERTAIN", 45, 145, 400, 420);
  addPanel(s, "Evaluation use", "The reference claim universe supports conservative matching against recorded system responses.\n\nIt is not official BBQ-V ground truth and it is not an LLM judge.", 515, 145, 400, 420);
}
{
  const s = resetCanvas(17, "Implementation");
  addText(s, "The project records completed model runs and evaluates them without new generation or external calls.", 80, 125, 800, 65, 24, "000000", true, "center");
  addPanel(s, "Core pipeline", "src/pipeline\n\nGeneration\nClaim extraction\nEvidence checking\nBias monitoring\nCorrection and verification", 55, 235, 395, 300);
  addPanel(s, "Evaluation artifacts", "scripts/evaluate_manual_reference.py\n\nmanual_reference_evaluation.json\nmanual_reference_evaluation.csv\nmanual_reference_evaluation_report.md", 510, 235, 395, 300);
}
{
  const s = resetCanvas(18, "Experiment Coverage");
  addPanel(s, "Recorded systems", "vlm_only\n\ngeneric_self_correction\n\nproposed", 45, 160, 395, 350);
  addPanel(s, "Reference set", "3 BBQ-V images\n25 manually reviewed reference claims\n\n9 completed experiment outputs\n\nMaximum evaluation size: 10 images", 515, 160, 395, 350);
  addText(s, "The current evaluation uses only outputs already recorded locally.", 80, 565, 800, 42, 21, "000000", true, "center");
}

await fs.mkdir(path.dirname(candidatePath), { recursive: true });
await (await PresentationFile.exportPptx(p)).save(candidatePath);

const requirements = {
  explicitTotalSlideCount: 22,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
};
await fs.mkdir(path.dirname(finalPath), { recursive: true });
await finalizePresentation({
  ...requirements,
  workspaceDir,
  candidatePath,
  finalPath,
  pythonExecutable: "C:/Users/gagan/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe",
  integrityValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: ["--expected-slide-size-emu", "9144000,6858000", "--validate-bullet-geometry", "--validate-heading-fit"],
  verifyArtifactToolImport: true,
  receiptPath: path.join(tmpDir, "multimodal_bias_correction.validation.json"),
});
