import fs from "node:fs/promises";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";
const source = "C:/Users/gagan/multimodal-bias-correction/output/multimodal_bias_correction_project.pptx";
const p = await PresentationFile.importPptx(await FileBlob.load(source));
const montage = await p.export({ format: "png", montage: true, scale: 1 });
await fs.writeFile("C:/Users/gagan/multimodal-bias-correction/.ppt_build/final-montage.png", new Uint8Array(await montage.arrayBuffer()));
for (let i = 0; i < p.slides.items.length; i += 1) {
  const png = await p.slides.getItem(i).export({ format: "png", scale: 1 });
  await fs.writeFile(`C:/Users/gagan/multimodal-bias-correction/.ppt_build/final-slide-${i + 1}.png`, new Uint8Array(await png.arrayBuffer()));
}
const inspect = await p.inspect({ kind: "slide,textbox,shape,image,notes", maxChars: 16000 });
await fs.writeFile("C:/Users/gagan/multimodal-bias-correction/.ppt_build/final-inspect.ndjson", inspect.ndjson);
