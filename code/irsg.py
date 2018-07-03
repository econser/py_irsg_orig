# scene graph -> factor graph
# run GOP
# score each box (objects)
# score each box (attrs)
# score each box pair (relationships)
# infer energy
from gop import *

def get_energy(query, image):
    boxes = get_boxes(image)



def get_boxes(image):
    #from pylab import *
    import numpy as np
    #from util import *
    
    n_seeds = 140
    segmentations_per_seed = 4
    max_iou = 0.8
    seed_method = setupLearned(n_seeds, segmentations_per_seed, max_iou)
    prop = proposals.Proposal(seed_method)
    
    detector = contour.MultiScaleStructuredForest()
    detector.load( "./sf.dat" )
    
    s = segmentation.geodesicKMeans(imgproc.imread(image), detector, 1000)
    b = prop.propose(s)
    
    boxes = s.maskToBox( b )
    print("Generated %d proposals".format(b.shape[0]))
    
    figure()
    for i in range(min(20,b.shape[0])):
        im = np.array( s.image )
        im[ b[i,s.s] ] = (255,0,0)
        ax = subplot( 4, 5, i+1 )
        ax.imshow( im )
        # Draw the bounding box
        from matplotlib.patches import FancyBboxPatch
        ax.add_patch( FancyBboxPatch( (boxes[i,0],boxes[i,1]), boxes[i,2]-boxes[i,0], boxes[i,3]-boxes[i,1], boxstyle="square,pad=0.", ec="b", fc="none", lw=2) )
    show()
