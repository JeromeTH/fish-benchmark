from fish_benchmark.data.abby import AbbyAnnotation
PATH = '/share/j_sun/jth264/abby'

if __name__ == '__main__':
    annotation = AbbyAnnotation(PATH)
    for track, ann in annotation.stream_annotations():
        print(track)
        print(ann)
        break