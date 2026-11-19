import { FileBlob, PresentationFile } from "@oai/artifact-tool";
const p = await PresentationFile.importPptx(await FileBlob.load("C:/Users/gagan/Downloads/PPT.pptx"));
const s = p.slides.getItem(3);
for (const item of [s, s.images.items[0], s.shapes.items[0]]) {
  console.log(Object.getPrototypeOf(item).constructor.name, Object.getOwnPropertyNames(Object.getPrototypeOf(item)));
}
