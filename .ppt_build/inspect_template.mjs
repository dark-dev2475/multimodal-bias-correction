import fs from "node:fs/promises";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const source = "C:/Users/gagan/Downloads/PPT.pptx";
const presentation = await PresentationFile.importPptx(await FileBlob.load(source));
const snapshot = await presentation.inspect({
  kind: "deck,slide,textbox,shape,image,table,chart,notes,layout",
  maxChars: 120000,
});
await fs.writeFile("C:/Users/gagan/multimodal-bias-correction/.ppt_build/template-inspect.ndjson", snapshot.ndjson);
const montage = await presentation.export({ format: "png", montage: true, scale: 1 });
await fs.writeFile("C:/Users/gagan/multimodal-bias-correction/.ppt_build/template-montage.png", new Uint8Array(await montage.arrayBuffer()));
for (let index = 0; index < presentation.slides.items.length; index += 1) {
  const image = await presentation.slides.getItem(index).export({ format: "png", scale: 1 });
  await fs.writeFile(`C:/Users/gagan/multimodal-bias-correction/.ppt_build/template-slide-${index + 1}.png`, new Uint8Array(await image.arrayBuffer()));
}
