import tensorflow as tf
import io
import cv2
count=0
def write_on_tf(img, label, writer):
    global count
    count+=1
    print(count)
    encoded_jpg_io = io.BytesIO(img)
    label = write_age(label)

    example = tf.train.Example(features=tf.train.Features(feature={
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_jpg_io.getvalue()])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label))}))

    writer.write(example.SerializeToString())

def save_and_write(tfwriter, filename, label, image):
    #cv2.imwrite(filename, image)
    #cv2.imshow('cazziemazzi',image)
    #cv2.waitKey(0)

    write_on_tf(image, label, tfwriter)

def write_age(n):
    age = []
    for i in range(0, 101):
        if i != n:
            age.append(0)
        else:
            age.append(1)
    return age