from __future__ import print_function
import data_pull as dp

import json
cfg_file = open('config.json')
cfg_data = json.load(cfg_file)
out_path = cfg_data['file_paths']['output_path']
img_path = cfg_data['file_paths']['image_path']
mat_path = cfg_data['file_paths']['mat_path']

"""
import numpy as np
np.set_printoptions(suppress=True)

import data_pull as dp
vgd, potentials, platt_mod, bin_mod, queries, ifdata = dp.get_all_data()

import geo_mean_check as gm
reload(gm); gm.go(queries, potentials, platt_mod)

import image_fetch_plot as ifp
import image_fetch_utils as ifu
data_gm = [('/home/econser/School/Thesis/data/batches/gm_simple/', 'geo_mean'), ('/home/econser/School/Thesis/data/batches/orig_simple/', 'original')]
tp_simple = ifu.get_partial_scene_matches(vgd['vg_data_test'], queries['simple_graphs'])
ifp.r_at_k_plot_simple(data_gm, tp_simple)



base_path = '/home/econser/School/Thesis/sg_dataset/sg_test_images/'
img_filenames = []
for img in vgd['vg_data_test']:
    img_name = img.image_path.split('/')[-1]
    img_filenames.append(base_path + img_name)

scores, best_match, top_patches = gm.get_scores(np.arange(1000), queries['simple_graphs'][0].annotations, 0, potentials, platt_mod)

gm.draw_patches(img_filenames[222], best_match[222], top_patches[222])



import cPickle
f = open('/home/econser/School/top_patches_test_q0.csv', 'wb')
cPickle.dump(top_patches, f, cPickle.HIGHEST_PROTOCOL)
f.close()
"""
def go(queries, potentials, platt_models):
    import sys
    import time
    import numpy as np
    
    image_ixs = np.arange(1000)
    base_path = '/home/econser/School/Thesis/data/batches/gm_simple/'
    
    total_time = 0
    for q_ix, query in enumerate(queries['simple_graphs']):
        avg_time = total_time / ((q_ix + 1) * 1.)
        remaining_time = (150 - q_ix) * avg_time
        status =  'query: {:03d} - total time: {:.2f} sec (avg: {:.2f}, remaining: {:.2f})'.format(q_ix, total_time, avg_time, remaining_time)
        print(status, end='\r'); sys.stdout.flush()
        
        start = time.time()
        scores, viz_data = get_scores(image_ixs, query.annotations, q_ix, potentials, platt_models, base_path)
        total_time += time.time() - start




"""
vgd, potentials, platt_mod, bin_mod, queries, ifdata = dp.get_all_data()
scores, best_match, top_patches = gm.get_scores(np.arange(1000), queries['simple_graphs'][0].annotations, 0, potentials, platt_mod)
"""
def get_scores(image_ixs, query_obj, query_ix, potentials, platt_models, output_path=''):
    import cPickle
    import numpy as np
    import image_fetch_utils as ifu
    
    object_names = []
    for obj in query_obj.objects:
        object_names.append(obj.names)
    
    image_scores = []
    viz_data = []
    top_box_data = []
    
    for image_ix in image_ixs:
        object_probs = []
        bbox_list = []
        top_list = []
        obj_detections = ifu.get_object_detections(image_ix, potentials, platt_models)
        for obj_name in object_names:
            obj_key = 'obj:' + obj_name
            if not obj_detections.has_key(obj_key):
                import pdb; pdb.set_trace()
                continue
            object_probs.append(obj_detections[obj_key][:,4])
            
            sort_ixs = np.argsort(obj_detections[obj_key][:,4])
            
            best_bbox_ix = sort_ixs[::-1][0]
            best_bbox = obj_detections[obj_key][best_bbox_ix, 0:4]
            bbox_tup = (obj_name, best_bbox)
            bbox_list.append(bbox_tup)
            
            top_n = sort_ixs[::-1][:10]
            top_bboxes = obj_detections[obj_key][top_n, 0:4]
            top_scores = obj_detections[obj_key][top_n, 4]
            top_tup = (obj_name, top_bboxes, top_scores)
            top_list.append(top_tup)
        object_probs = np.array(object_probs)
        viz_data.append(bbox_list)
        top_box_data.append(top_list)
        
        top_object_ixs = np.argmax(object_probs, axis=1)
        top_probs = []
        for i, probs in enumerate(object_probs):
            top_probs.append(probs[top_object_ixs[i]])
        top_probs = np.array(top_probs)
        gmean = top_probs.prod()**(1.0/len(top_probs))
        image_scores.append((image_ix, np.exp(-gmean)))
    
    if output_path != '':
        filename = 'q{:03d}_energy_values.csv'.format(query_ix)
        fq_filename = '{}{}'.format(output_path, filename)
        np.savetxt(fq_filename, image_scores, delimiter=',', header='image_ix, energy', fmt='%d, %3.4f', comments='')
        
        filename = 'q{:03d}_best_bboxes.csv'.format(query_ix)
        fq_filename = '{}{}'.format(output_path, filename)
        f = open(fq_filename, 'wb')
        cPickle.dump(viz_data, f, cPickle.HIGHEST_PROTOCOL)
        f.close()
        
        filename = 'q{:03d}_top10_bboxes.csv'.format(query_ix)
        fq_filename = '{}{}'.format(output_path, filename)
        f = open(fq_filename, 'wb')
        cPickle.dump(top_box_data, f, cPickle.HIGHEST_PROTOCOL)
        f.close()
    
    return np.array(image_scores), viz_data, top_box_data



def viz(image_filename, obj_box_pairs, color_list=None, title="", output_filename="", verbose=False):
    import os.path
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.patheffects as path_effects
    
    if not os.path.isfile(image_filename):
        return
    
    if color_list is None:
        import randomcolor
        rc = randomcolor.RandomColor()
        colorset = rc.generate(luminosity='bright', count=len(obj_box_pairs), format_='rgb')
        color_list = []
        for i in range(0, len(obj_box_pairs)):
            color_array = np.fromstring(colorset[i][4:-1], sep=",", dtype=np.int)
            color_array = color_array * (1. / 255.)
            color_list.append(color_array)
    
    img = Image.open(image_filename)
    img_array = np.array(img, dtype=np.uint8)
    fig, ax = plt.subplots(1)
    ax.imshow(img_array)
    
    for ix, obj_and_box in enumerate(obj_box_pairs):
        box = obj_and_box[1]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        
        box = patches.Rectangle((x,y),w,h, linewidth=4, edgecolor=color_list[ix], facecolor='none')
        ax.add_patch(box)
        txt = ax.text(x+5, y+5, obj_and_box[0], va='top', size=16, weight='bold', color='0.1')
        txt.set_path_effects([path_effects.withStroke(linewidth=1, foreground='w')])
    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    plt.tight_layout(pad=7.5)
    
    if len(output_filename) == 0:
        #plt.show(bbox_inches='tight')
        plt.show()
    else:
        plt.rcParams.update({'font.size': 10})
        plt.savefig(output_filename, dpi=175)
    
    plt.clf()
    plt.close()



def draw_patches(image_filename, best_boxes, top_class_boxes, color_list=None, max_width=200, max_height=200):
    import os.path
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.gridspec import GridSpec
    
    if not os.path.isfile(image_filename):
        return
    
    if color_list is None:
        import randomcolor
        rc = randomcolor.RandomColor()
        colorset = rc.generate(luminosity='bright', count=len(best_boxes), format_='rgb')
        color_list = []
        for i in range(0, len(best_boxes)):
            color_array = np.fromstring(colorset[i][4:-1], sep=",", dtype=np.int)
            color_array = color_array * (1. / 255.)
            color_list.append(color_array)
    
    n_classes = len(top_class_boxes)
    n_patches = len(top_class_boxes[1][1])
    
    # set up the layout
    gs = GridSpec(n_patches, n_patches)
    
    # put the image on the left side, right cols for the patches
    fig = plt.figure()
    img = Image.open(image_filename)
    img_a = np.array(img, dtype=np.uint8)
    
    ax = plt.subplot(gs[0:n_patches, 0:(n_patches-n_classes)])
    ax.imshow(img_a)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # draw the best box indicators
    for ix, obj_and_box in enumerate(best_boxes):
        box = obj_and_box[1]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        
        box = patches.Rectangle((x,y),w,h, linewidth=4, edgecolor=color_list[ix], facecolor='none')
        ax.add_patch(box)
        #txt = ax.text(x+5, y+5, obj_and_box[0], va='top', size=16, weight='bold', color='0.1')
        #txt.set_path_effects([path_effects.withStroke(linewidth=1, foreground='w')])
    
    # draw the patches next to the image
    for col_num, box_list in enumerate(top_class_boxes):
        for row_num, box in enumerate(box_list[1]):
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            
            img_copy = img.copy()
            img_crop = img_copy.crop((x, y, x+w, y+h))
            resize_ratio = min(max_width / w, max_height / h)
            new_w = int(w * resize_ratio)
            new_h = int(h * resize_ratio)
            img_resize = img_crop.resize((new_w, new_h), Image.ANTIALIAS)
            crop_a = np.array(img_resize, dtype=np.uint8)
            
            ax = plt.subplot(gs[row_num, col_num+(n_patches - n_classes)])
            ax.imshow(crop_a)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    
    # draw it and clean up
    plt.tight_layout()
    plt.show()
    plt.clf()
    plt.close()



def xform_p(base_dir):
    import os.path
    import numpy as np
    k_count = np.zeros(1000)
    n_queries = 1000
    file_suffix = '_energy_values'
    for i in range(0, n_queries):
        filename = base_dir + 'q{:03d}'.format(i) + file_suffix + '.csv'
        if not os.path.isfile(filename):
            continue
        energies = np.genfromtxt(filename, delimiter=',', skip_header=1)
        energies[:,1] = np.exp(-energies[:,1])
        np.savetxt(filename, energies, delimiter=',', header='image_ix, energy', fmt='%d, %3.4f', comments='')
"""
np.argsort(scores[:,1])[:10]
q0 best - [222,  11,  56, 347,  77, 248, 741, 394, 468, 414]

np.argsort(scores[:,1])[::-1][:10]
q0 wrst - [232, 773, 902, 899, 272, 258, 621, 102, 714, 134]
"""
