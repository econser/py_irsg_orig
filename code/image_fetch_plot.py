import os.path
import randomcolor
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects



"""
data_full = [('/home/econser/School/Thesis/data/batches/orig_full/', 'original'), ('/home/econser/School/Thesis/data/batches/unif_full/', 'uniform'), ('/home/econser/School/Thesis/data/batches/emp_full/', 'empirical')]
tp_full = []
for i in range(0, 500):
  tp_full.append([i])
ifp.r_at_k_plot_simple(data_full, tp_full)

data_simple = [('/home/econser/School/Thesis/data/batches/orig_simple/', 'original'), ('/home/econser/School/Thesis/data/batches/unif_simple/', 'uniform'), ('/home/econser/School/Thesis/data/batches/emp_simple/', 'empirical')]
ifp.r_at_k_plot_simple(data_simple, tp_simple)

data_obscured = [('/home/econser/School/Thesis/data/batches/orig_obscure/', 'original'), ('/home/econser/School/Thesis/data/batches/unif_obscure/', 'uniform'), ('/home/econser/School/Thesis/data/batches/emp_obscure/', 'empirical')]
tp_simple = ifu.get_partial_scene_matches(vgd['vg_data_test'], queries['simple_graphs'])
ifp.r_at_k_plot_simple(data_obscured, tp_simple)
"""
def r_at_k_plot_simple(data, ground_truth_map, do_holdout=False, x_limit=-1):
  import image_fetch_utils as ifu; reload(ifu)
  plot_handles = []
  
  plt.figure(1)
  plt.grid(True)
  
  for path_tup in data:
    vals = ifu.get_r_at_k_simple(path_tup[0], ground_truth_map, do_holdout)
    plot_handle, = plt.plot(np.arange(len(vals)), vals, label=path_tup[1])
    plot_handles.append(plot_handle)
  
  plt.xlabel("k")
  plt.ylabel("Recall at k")
  
  plt.legend(handles=plot_handles, loc=4)
  if x_limit > 0:
    plt.xlim([0, x_limit])
  plt.ylim([0, 1])
  
  plt.show()



def k_chart(csv_path, query_details):
  """ plot k-value vs various metrics
    q_details: index, n_objects, rel_per_obj, n_attr, n_rel
  """
  import glob
  from parse import parse
  
  file_list = glob.glob(csv_path+'*.csv')
  en_list = []
  for f in file_list:
    file = parse(csv_path+'{}', f)[0]
    ix = int(parse('q{}_energy_values.csv', file)[0])
    
    csv_data = np.genfromtxt(csv_path+file, delimiter=',')
    sort_ix = np.argsort(csv_data[:,1])
    k = np.where(csv_data[sort_ix][:,0] == ix)
    
    q_ix = np.where(query_details[:,0] == ix)
    n_obj = query_details[q_ix, 1]
    rpo = query_details[q_ix, 2]
    
    en_list.append((ix, k[0][0], n_obj[0][0], np.round(rpo[0][0], decimals=2)))
  data = np.array(en_list, dtype=object)

  plt.figure(1)
  plt.subplot(121)
  plt.grid(b=True)
  plt.scatter(data[:,2], data[:,1])
  plt.xlabel('object count')
  
  plt.subplot(122)
  plt.grid(b=True)
  plt.scatter(data[:,3], data[:,1])
  plt.xlabel('relationships per object')
  
  plt.show()
  
  plt.clf()
  plt.close()



def p_compare(tracker, detections, marginals, filename=''):
  rc = randomcolor.RandomColor()
  n_objects = len(tracker.object_names)
  colorset = rc.generate(luminosity='bright', count=n_objects, format_='rgb')
  color_list = []
  for i in range(0, n_objects):
    color_array = np.fromstring(colorset[i][4:-1], sep=",", dtype=np.int)
    color_array = color_array * (1. / 255.)
    color_list.append(color_array)
  
  plt.figure(1)
  plt.grid(b=True)
  legend_list = []
  max_x = 0.0
  max_y = 0.0
  for obj_ix in range(0, n_objects):
    obj_name = tracker.object_names[obj_ix]
    dets = detections['obj:'+obj_name]
    detection_probs = dets[:,4]
    
    det_sum = np.sum(detection_probs)
    norm_dets = detection_probs / det_sum
    if np.max(norm_dets) > max_x: max_x = np.max(norm_dets)
    
    marginal_sum = np.sum(marginals[obj_ix])
    norm_marginal = marginals[obj_ix] / marginal_sum
    if np.max(norm_marginal) > max_y: max_y = np.max(norm_marginal)
    
    sctr_plt = plt.scatter(norm_dets, norm_marginal, color = color_list[obj_ix])
    legend_list.append(sctr_plt)

  min_of_max = min(max_x, max_y)
  plt.plot([0., min_of_max], [0., min_of_max])
  
  plt.ylabel('marginals')
  plt.xlabel('detections')
  
  plt.legend(legend_list, tracker.object_names, scatterpoints=1)

  if len(filename) > 0:
    plt.savefig(filename)
  else:
    plt.show()
  
  plt.clf()
  plt.close()



def draw_heatmap(image_filename, box_list, detection_values, marginal_values=[], method="", rcnn_detect=False, object_name="", filename=""):
  from scipy.ndimage.filters import gaussian_filter
  
  # open the image
  image_is_local = os.path.isfile(image_filename)
  if not image_is_local:
    return
  
  img = Image.open(image_filename).convert("L")
  img_array = np.array(img)
  
  do_marginals = False
  if len(marginal_values) > 0: do_marginals = True
  
  # generate the detections map
  img_width = (img_array.shape)[0]
  img_height = (img_array.shape)[1]
  d_map = np.zeros((img_height, img_width), dtype=np.float)
  m_map = np.zeros((img_height, img_width), dtype=np.float)
  delta_map = np.zeros((img_height, img_width), dtype=np.float)
  
  for i in range(0, len(box_list)):
    box = box_list[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    
    p_d = detection_values[i]
    d_map[x:x+w, y:y+h] = d_map[x:x+w, y:y+h] + p_d
    
    if do_marginals:
      p_m = marginal_values[i]
      m_map[x:x+w, y:y+h] = m_map[x:x+w, y:y+h] + p_m
  
  if do_marginals:
    plt.figure(1)
    plt.xticks([])
    plt.yticks([])
    #plt.title(object_name)
    m_map_blur = gaussian_filter(m_map, sigma=7)
    plt.imshow(img_array, cmap='gray')
    plt.imshow(m_map_blur.T, alpha=0.4)
    plt.tight_layout()
    
    if len(filename) > 0:
      plt.savefig(filename+'_{}_{}_mgnl.png'.format(object_name, method), dpi=175)
    else:
      plt.show()
    
    plt.clf()
    plt.close()
  
  plt.figure(1)
  plt.xticks([])
  plt.yticks([])
  #if rcnn_detect:
  #  plt.title(object_name)
  #else:
  #  plt.title(object_name + ' (sub)')
  plt.imshow(img_array, cmap='gray')
  d_map_blur = gaussian_filter(d_map, sigma=7)
  plt.imshow(d_map_blur.T, alpha=0.4)
  plt.tight_layout()
    
  if len(filename) > 0:
    if rcnn_detect:
      plt.savefig(filename+'_{}_{}_dets.png'.format(object_name, method), dpi=175)
    else:
      plt.savefig(filename+'_{}_{}_dets_sub.png'.format(object_name, method), dpi=175)
  else:
    plt.show()
    
  plt.clf()
  plt.close()






def draw_all_heatmaps(tracker, detections, marginals, method, output_path = './', image_prefix = ''):
  for obj_ix in range(0, len(tracker.object_names)):
    obj_name = tracker.object_names[obj_ix]
    dets = tracker.unary_detections
    out_filename = image_prefix
    draw_heatmap(tracker.image_path, tracker.box_coords, dets[obj_ix], marginals[obj_ix], method, rcnn_detect=tracker.detected_vars[obj_ix], object_name=obj_name, filename=output_path + out_filename)



def draw_each_box(filename, detections, do_heatmap=False):
  #plt.switch_backend('Qt4Agg')
  
  image_is_local = os.path.isfile(filename)
  if not image_is_local:
    return
  
  img = Image.open(filename)
  img_array = np.array(img, dtype=np.uint8)

  n_boxes = len(detections)
  for i in range(0, 3):
    fig, ax = plt.subplots(1)
    ax.imshow(img_array)
    
    box = detections[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    p = box[4]
    
    title = "b{0}: {1}, {2}, {3}, {4} -> {5}".format(i, x, y, w, h, p)
    
    if do_heatmap:
      img_width = (img_array.shape)[0]
      img_height = (img_array.shape)[1]
      heat = np.zeros((img_height, img_width), dtype=np.float)
      heat[x:x+w, y:y+h] = heat[x:x+w, y:y+h] + 1.
      plt.imshow(heat.T, cmap='bone', alpha=0.6)
    else:
      box = patches.Rectangle((x,y),w,h, linewidth=3, edgecolor='red', facecolor='none')
      ax.add_patch(box)
    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    plt.title(title)
    
    plt.tight_layout(pad=1.25)
    plt.show()
    
    plt.clf()
    plt.close()




def draw_best_objects(tracker, indices, energy=None, filename="", image_size=[], gen_title=False):
  n_objects = len(tracker.object_names)
  rc = randomcolor.RandomColor()
  colorset = rc.generate(luminosity='bright', count=n_objects, format_='rgb')
  color_list = []
  for i in range(0, n_objects):
    color_array = np.fromstring(colorset[i][4:-1], sep=",", dtype=np.int)
    color_array = color_array * (1. / 255.)
    color_list.append(color_array)
  
  legend_list = []
  for i in range(0, n_objects):
    legend_list.append((tracker.object_names[i], color_list[i]))
  
  box_list = []
  for i in range(0, n_objects):
    box_coords = tracker.box_pairs[indices[i]][1][0:4]
    box_and_color = np.hstack((box_coords, color_list[i]))
    box_list.append(box_and_color)
  
  title = ""
  if gen_title:
    title = "Object Detections"
    
    if energy != None:
      title = "Object Detections (energy={0:.3f})".format(energy)
  draw_image_box(tracker.image_path, box_list, legend_list, title, filename, size=image_size)



def draw_image_box(image_path, box_list, legend_list=[], relation_name="", filename="", size=[], verbose=False):
  #plt.switch_backend('Qt4Agg')
  
  if verbose: print '  dib: checking for image file\r',
  image_is_local = os.path.isfile(image_path)
  if not image_is_local:
    return
  
  if verbose: print '  dib: opening image\r',
  img = Image.open(image_path)
  img_array = np.array(img, dtype=np.uint8)
  fig, ax = plt.subplots(1)
  ax.imshow(img_array)
  
  if verbose: print '  dib: adding rectangles\r',
  for i in range(0, len(box_list)):
    box = box_list[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    c = box[4:7]
    box = patches.Rectangle((x,y),w,h, linewidth=4, edgecolor=c, facecolor='none')
    ax.add_patch(box)
    txt = ax.text(x+5, y+5, legend_list[i][0], va='top', size=16, weight='bold', color='0.1')
    txt.set_path_effects([path_effects.withStroke(linewidth=1, foreground='w')])
  
  if verbose: print '  dib: removing axes\r',
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  
  if verbose: print '  dib: adding title\r',
  if len(relation_name) > 0:
    plt.title(relation_name)
  
  if verbose: print '  dib: adding legend\r',
  handle_list = []
  name_list = []
  for i in range(0, len(legend_list)):
    h = plt.Rectangle((0,0),0.125,0.125, fc=legend_list[i][1])
    handle_list.append(h)
    name_list.append(legend_list[i][0])
  #plt.legend(handle_list, name_list, bbox_to_anchor=(1.14, 1.01))#, loc='upper right')
  
  plt.tight_layout(pad=7.5)
  
  if verbose: print '  dib: showing plot\r'
  if len(filename) == 0:
    #plt.show(bbox_inches='tight')
    plt.show()
  else:
    plt.rcParams.update({'font.size': 10})
    plt.savefig(filename, dpi=175)
  
  plt.clf()
  plt.close()



def draw_best_relationships(n_pairs, tracker, rel_num, verbose=False):
  image_path = tracker.image_path
  box_pairs = tracker.box_pairs
  ll_relationships = tracker.relationships[rel_num].ll_relationship
  object_boxes = tracker.relationships[rel_num].object_boxes
  object_name = tracker.relationships[rel_num].object_name
  subject_boxes = tracker.relationships[rel_num].subject_boxes
  subject_name = tracker.relationships[rel_num].subject_name
  relation_name = tracker.relationships[rel_num].relationship

  if verbose: print '  dbr: generating scores\r',
  model_class_scores = np.array([x for x in itertools.product(subject_boxes[:,4], object_boxes[:,4])])
  model_class_scores = -np.log(model_class_scores)
  composite_scores = np.zeros(model_class_scores.shape[0])
  ll_vals = np.concatenate((model_class_scores, ll_relationships[:,np.newaxis], composite_scores[:,np.newaxis]), axis=1)
  ll_vals[:,3] = np.sum(ll_vals[:,0:4], axis=1)
  sorted_indices = np.argsort(1.0 * ll_vals[:,3])

  if verbose: print '  dbr: creating box list\r',
  box_list = []
  for i in range(0, n_pairs):
    j = sorted_indices[i]
    ms = ModelState(box_pairs[j, 0, 0:4], box_pairs[j, 1, 0:4], ll_vals[j, 0], ll_vals[j, 1], ll_vals[j, 2], ll_vals[j, 3])
    box_list.append(ms)

  if verbose: print '  dbr: generating colors\r',
  sub_color = np.array([1., 0., 0.])
  obj_color = np.array([0., 0., 1.])

  if verbose: print '  dbr: prepping boxes for drawing\r',
  best_boxes = []
  for i in range(0, n_pairs):
    best_boxes.append(np.hstack((box_list[i].subject_box, np.copy(sub_color))))
    best_boxes.append(np.hstack((box_list[i].object_box, np.copy(obj_color))))

  if verbose: print '  dbr: generating legend entries\r',
  legend_list = []
  legend_list.append((subject_name, obj_color))
  legend_list.append((object_name, sub_color))

  if verbose: print '  dbr: calling draw_image_box\r',
  draw_image_box(image_path, best_boxes, legend_list, relation_name, verbose)



def draw_gm(gm, layout='sfdp', size=1.5):
  import opengm as ogm
  import networkx
  from networkx.drawing.nx_agraph import graphviz_layout
  networkx.graphviz_layout = graphviz_layout

  ogm.visualizeGm(gm, layout=layout, relNodeSize=size)



# def draw_heatmap_old(image_filename, box_list, detection_values, marginal_values=[], plot_deltas=False, title="", filename=""):
#   # open the image
#   image_is_local = os.path.isfile(image_filename)
#   if not image_is_local:
#     return

#   img = Image.open(image_filename).convert("L")
#   img_array = np.array(img)

#   do_marginals = False
#   if len(marginal_values) > 0: do_marginals = True

#   # generate the detections map
#   img_width = (img_array.shape)[0]
#   img_height = (img_array.shape)[1]
#   d_map = np.zeros((img_height, img_width), dtype=np.float)
#   m_map = np.zeros((img_height, img_width), dtype=np.float)
#   delta_map = np.zeros((img_height, img_width), dtype=np.float)

#   for i in range(0, len(box_list)):
#     box = box_list[i]
#     x = box[0]
#     y = box[1]
#     w = box[2]
#     h = box[3]

#     p_d = detection_values[i]
#     d_map[x:x+w, y:y+h] = d_map[x:x+w, y:y+h] + p_d

#     if do_marginals:
#       p_m = marginal_values[i]
#       m_map[x:x+w, y:y+h] = m_map[x:x+w, y:y+h] + p_m

#   if do_marginals:
#     plt.figure(1)
#     plt.suptitle(title)

#     dets_pid = 121
#     delta_pid = 0
#     marginal_pid = 122
#     if plot_deltas:
#       dets_pid = 131
#       delta_pid = 132
#       marginal_pid = 133

#     plt.subplot(dets_pid)
#     plt.imshow(img_array, cmap='gray')
#     plt.imshow(d_map.T, alpha=0.3)
#     plt.title("DETECTIONS")
#     plt.xticks([])
#     plt.yticks([])

#     if plot_deltas:
#       plt.subplot(delta_pid)
#       plt.imshow(img_array, cmap='gray')
#       plt.imshow((m_map / np.amax(m_map) - d_map / np.amax(d_map)).T, alpha=0.3)
#       plt.title("DELTA")
#       plt.xticks([])
#       plt.yticks([])

#     plt.subplot(marginal_pid)
#     plt.imshow(img_array, cmap='gray')
#     plt.imshow(m_map.T, alpha=0.3)
#     plt.title("MARGINALS")
#     plt.xticks([])
#     plt.yticks([])
#   else:
#     plt.figure(1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.title("DETECTIONS")
#     plt.imshow(img_array)
#     plt.imshow(d_map.T, alpha=0.3)

#   plt.tight_layout()
#   if len(filename) > 0:
#     #plt.switch_backend('Agg')
#     plt.savefig(filename, dpi=175)
#   else:
#     #plt.switch_backend('Qt4Agg')
#     plt.show()

#   plt.clf()
#   plt.close()
