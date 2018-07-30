from pycocotools.coco import COCO
from pycocotools.mask import frPyObjects, decode
import cv2
import numpy as np
import tensorflow as tf


class dataset:
    def __init__(self, cocoImagesDir, cocoAnnotationsFile):
        self.cocoImagesDir = cocoImagesDir
        self.cocoAnnotationsFile = cocoAnnotationsFile
        self.coco = COCO(cocoAnnotationsFile)

    def getImgIdsFromClassNames(self, classNames=None):
        catIds = self.coco.getCatIds(catNms=classNames)
        return self.coco.getImgIds(catIds=catIds)

    def getAnnotationsFromImgIds(self, imgIds, classNames=None):
        if classNames is not None:
            catIds = self.coco.getCatIds(catNms=classNames)
            annIds = self.coco.getAnnIds(imgIds=imgIds, catIds=catIds)
        else:
            annIds = self.coco.getAnnIds(imgIds=imgIds)
        return self.coco.loadAnns(ids=annIds)

    def getImagesFromImgIds(self, imgIds):
        for img in self.coco.loadImgs(ids=imgIds):
            yield cv2.imread("{}/{}".format(self.cocoImagesDir, img["file_name"]))

    def getClassNames(self):
        return [c["name"] for c in self.coco.dataset['categories']]

    def getBinaryMaskFromImgId(self, imgId, classNames=None):
        img = next(self.getImagesFromImgIds(imgId))
        anns = self.getAnnotationsFromImgIds(imgId, classNames)
        mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=bool)
        for ann in anns:
            classMasks = decode(frPyObjects(ann["segmentation"], mask.shape[0], mask.shape[1]))
            if classMasks.ndim > 2 and classMasks.shape[2] > 1:
                classMask = np.sum(classMasks, axis=2, keepdims=True)
            else:
                classMask = classMasks if classMasks.ndim > 2 else classMasks[..., np.newaxis]
            mask = np.logical_or(mask, classMask > 0)
        return mask.astype(np.uint8)

    def CreateCocoTFRecord(self, outFile, imgIds=None, classNames=None):
        #https://planspace.org/20170427-sparse_tensors_and_tfrecords/
        if imgIds is None:
            imgIds = self.getImgIdsFromClassNames(classNames=classNames)
        writer = tf.python_io.TFRecordWriter(outFile)
        for imgId in imgIds:
            image = next(self.getImagesFromImgIds(imgId))[..., ::-1]
            mask = self.getBinaryMaskFromImgId(imgId, classNames)
            features = {
                "imageContent": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tostring()])),
                "maskContent": tf.train.Feature(bytes_list=tf.train.BytesList(value=[mask.tostring()])),
                "height": tf.train.Feature(int64_list=tf.train.Int64List(value=[image.shape[0]])),
                "width": tf.train.Feature(int64_list=tf.train.Int64List(value=[image.shape[1]])),
            }
            example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(example.SerializeToString())
        writer.close()


ds = dataset("/Users/diogoc/Downloads/coco/val2017",
             "/Users/diogoc/Downloads/coco/annotations/instances_val2017.json")
# for className in ds.getClassNames():
#     print(className)
classNames = ["tennis racket", "person"]
imgIds = ds.getImgIdsFromClassNames(classNames)
print("{} images found.".format(len(imgIds)))
# for imgId in imgIds:
#     img = next(ds.getImagesFromImgIds(imgId))
#     binMask = ds.getBinaryMaskFromImgId(imgId, classNames)
#     cv2.imshow('maskedImage', img*binMask)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
ds.CreateCocoTFRecord("/Users/diogoc/Downloads/coco/val2017.tfrecord", imgIds, classNames)

