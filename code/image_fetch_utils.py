import image_fetch_core as ifc
import numpy as np
import scipy.io as sio
import os.path

class RelationshipParameters (object):
  def __init__(self, platt_a, platt_b, gmm_weights, gmm_mu, gmm_sigma):
    self.platt_a = platt_a
    self.platt_b = platt_b
    self.gmm_weights = gmm_weights
    self.gmm_mu = gmm_mu
    self.gmm_sigma = gmm_sigma

#===============================================================================
# MAT FILE MANAGEMENT UTILITIES
#-------------------------------------------------------------------------------
# Functions for reading and manipulating the MATLAB .mat files into more
#  python-friendly structures
#===============================================================================
def get_mat_data(data_path="/home/econser/School/Thesis/code/model_params/"):
  """
  load the data files for use in running the model.
  expects the files to be all in the same directory

  Args:
    data_path: the fully-quaified path to the data files

  Returns:
    vgd (.mat file): vg_data file
    potentials (.mat file): potentials file
    platt_mod (.mat file): platt model file
    bin_mod (.mat file): GMM parameters
    queries (.mat file): queries
  """
  print("loading vg_data file...")
  vgd_path = data_path + "vg_data.mat"
  vgd = sio.loadmat(vgd_path, struct_as_record=False, squeeze_me=True)
  
  print("loading potentials data...")
  potentials_path = data_path + "potentials_s.mat"
  potentials = sio.loadmat(potentials_path, struct_as_record=False, squeeze_me=True)
  
  print("loading binary model data...")
  binary_path = data_path + "binary_models_struct.mat"
  bin_mod_mat = sio.loadmat(binary_path, struct_as_record=False, squeeze_me=True)
  bin_mod = get_relationship_models(bin_mod_mat)
  
  print("loading platt model data...")
  platt_path = data_path + "platt_models_struct.mat"
  platt_mod = sio.loadmat(platt_path, struct_as_record=False, squeeze_me=True)
  
  print("loading test queries...")
  query_path = data_path + "simple_graphs.mat"
  queries = sio.loadmat(query_path, struct_as_record=False, squeeze_me=True)
  
  return vgd, potentials, platt_mod, bin_mod, queries



def get_energy_metrics(image_ix, n_values, csv_path='/home/econser/School/Thesis/data/inference test/energies/'):
  import numpy as np
  filename = csv_path + 'q{0:03}_energy_values.csv'.format(image_ix)
  nrg = np.genfromtxt(filename, delimiter=',')
  sort_ix = np.argsort(nrg[:,1])
  low_energies = nrg[sort_ix][:n_values][:,0]
  high_energies = (nrg[sort_ix][-(n_values+1):])[:n_values][:,0]
  ground_ix = np.where(nrg[sort_ix][:,0] == image_ix)
  return low_energies, high_energies, ground_ix



"""
k_vals = ifu.get_k_values('/home/econser/School/Thesis/data/default_energy_batch/')
import matplotlib as plt
plt.hist(k_vals[:,1], 50)
plt.show()
"""
def get_k_values(file_path):
  import numpy as np
  k_list = []
  for i in range(0, 999):
    filename = file_path + 'q{:03d}_energy_values.csv'.format(i)
    if not os.path.isfile(filename):
      continue
    energies = np.genfromtxt(filename, delimiter=',')
    sort_ix = np.argsort(energies[:,1])
    k = np.where(energies[sort_ix][:,0] == i)
    k_list.append((i, k[0][0]))
  return np.array(k_list)



def get_r_at_k(base_dir, file_suffix='_energy_values'):
  import numpy as np
  k_list = []
  for i in range(0, 999):
    filename = base_dir + 'q{:03d}'.format(i) + file_suffix + '.csv'
    if not os.path.isfile(filename):
      continue
    energies = np.genfromtxt(filename, delimiter=',')
    sort_ix = np.argsort(energies[:,1])
    k = np.where(energies[sort_ix][:,0] == i)
    k_list.append((i, k[0][0]))
  k_vals = np.array(k_list)
  n_at_k = []
  for k in range(0,len(k_vals)):
    n = len(np.where(k_vals[:,1] <= k)[0])
    n_at_k.append((k, n))
  r_at_k = np.array(n_at_k, dtype=np.float)
  r_at_k[:,1] = r_at_k[:,1] / n
  return r_at_k



def get_r_at_k_simple(base_dir, gt_map, do_holdout=False, file_suffix='_energy_values'):
  import numpy as np
  k_count = np.zeros(1000)
  n_queries = len(gt_map)
  for i in range(0, n_queries):
    filename = base_dir + 'q{:03d}'.format(i) + file_suffix + '.csv'
    if not os.path.isfile(filename):
      continue
    energies = np.genfromtxt(filename, delimiter=',', skip_header=1)
    sort_ix = np.argsort(energies[:,1])
    recall = np.ones(1000, dtype=np.float)
    for k in range(0, len(energies)):
      if do_holdout and energies[sort_ix][k][0] == gt_map[i][0]:
        recall[k] = 0.0
        continue
      indices = np.array(energies[sort_ix][0:k+1][:,0])
      true_positives = set(indices) & set(gt_map[i])
      if len(true_positives) > 0:
        break
      recall[k] = 0.0
    k_count += recall
  return k_count / n_queries



def get_r_at_k_simple_old(base_dir, gt_map, file_suffix='_energy_values'):
  import numpy as np
  k_count = np.zeros(1000)
  n_queries = len(gt_map)
  for i in range(0, n_queries):
    filename = base_dir + 'q{:03d}'.format(i) + file_suffix + '.csv'
    if not os.path.isfile(filename):
      continue
    energies = np.genfromtxt(filename, delimiter=',', skip_header=1)
    sort_ix = np.argsort(energies[:,1])
    recall = np.zeros(1000, dtype=np.float)
    #print '{:03d}'.format(i)
    for k in range(0, len(energies)):
      indices = np.array(energies[sort_ix][0:k+1][:,0])
      true_positives = set(indices) & set(gt_map[i])
      recall[k] = float(len(true_positives)) #/ len(gt_map[i])
      #print '   k:{:03d}  TP:{:03d}  POOL:{:03d}  ->  r@k{:0.3f}'.format(k, len(true_positives), len(gt_map[i]), recall[k])
    k_count += recall
  import pdb; pdb.set_trace()
  return k_count / n_queries



def gen_and_print(tp_full, tp_simple):
  k_vals = [1, 5, 10, 55, 60, 65]
  orig_full = get_data_table('orig_full', '/home/econser/School/Thesis/data/batches/orig_full/', tp_full, k_vals)
  emp_full = get_data_table('emp_full', '/home/econser/School/Thesis/data/batches/emp_full/', tp_full, k_vals)
  unif_full = get_data_table('unif_full', '/home/econser/School/Thesis/data/batches/unif_full/', tp_full, k_vals)
  
  orig_simple = get_data_table('orig_simple', '/home/econser/School/Thesis/data/batches/orig_simple/', tp_simple, k_vals)
  emp_simple = get_data_table('emp_simple', '/home/econser/School/Thesis/data/batches/emp_simple/', tp_simple, k_vals)
  unif_simple = get_data_table('unif_simple', '/home/econser/School/Thesis/data/batches/unif_simple/', tp_simple, k_vals)
  
  orig_obscured = get_data_table('orig_obscured', '/home/econser/School/Thesis/data/batches/orig_obscure/', tp_simple, k_vals)
  emp_obscured = get_data_table('emp_obscured', '/home/econser/School/Thesis/data/batches/emp_obscure/', tp_simple, k_vals)
  unif_obscured = get_data_table('unif_obscured', '/home/econser/School/Thesis/data/batches/unif_obscure/', tp_simple, k_vals)
  
  print_list = [orig_full, emp_full, unif_full, orig_simple, emp_simple, unif_simple, orig_obscured, emp_obscured, unif_obscured]
  for result in print_list:
    print '{}: {} {}'.format(result[0][1], result[1], result[2])
  
  return print_list


"""
orig_full = ifu.get_data_table('orig_full', '/home/econser/School/Thesis/data/batches/orig_full/', tp_full, [1, 5, 15, 55, 60, 65])
orig_simple = ifu.get_data_table('orig_simple', '/home/econser/School/Thesis/data/batches/orig_simple/', tp_simple, [1, 5, 15, 55, 60, 65])
orig_obscured = ifu.get_data_table('orig_obscured', '/home/econser/School/Thesis/data/batches/orig_obscure/', tp_simple, [1, 5, 15, 55, 60, 65])
emp_full = ifu.get_data_table('emp_full', '/home/econser/School/Thesis/data/batches/emp_full/', tp_full, [1, 5, 15, 55, 60, 65])
emp_simple = ifu.get_data_table('emp_simple', '/home/econser/School/Thesis/data/batches/emp_simple/', tp_simple, [1, 5, 15, 55, 60, 65])
emp_obscured = ifu.get_data_table('emp_obscured', '/home/econser/School/Thesis/data/batches/emp_obscure/', tp_simple, [1, 5, 15, 55, 60, 65])
unif_full = ifu.get_data_table('orig_full', '/home/econser/School/Thesis/data/batches/unif_full/', tp_full, [1, 5, 15, 55, 60, 65])
unif_simple = ifu.get_data_table('orig_simple', '/home/econser/School/Thesis/data/batches/unif_simple/', tp_simple, [1, 5, 15, 55, 60, 65])
unif_obscured = ifu.get_data_table('orig_obscured', '/home/econser/School/Thesis/data/batches/unif_obscure/', tp_simple, [1, 5, 15, 55, 60, 65])
"""
def get_data_table(series_name, data_path, tp_map, k_values):
  ret_list = []
  ret_list.append(("name", series_name))
  
  r_at_k_list = []
  r_at_k_data = r_at_k_table([(data_path,"")], tp_map)
  #r_at_k_data = r_at_k_data.T
  for val in k_values:
    r_at_k_list.append((val, np.round(r_at_k_data[val-1][1], decimals=3)))
  ret_list.append(("r_at_k", r_at_k_list))
  
  rank_data = rank_table(data_path, tp_map)
  ret_list.append(("median", np.median(rank_data)))
  ret_list.append(("std_dev", np.std(rank_data)))
  
  return ret_list



"""
full_data = ifu.r_at_k_table(data_full, tp_full)
obscured_data = ifu.r_at_k_table(data_obscured, tp_simple)
simple_data = ifu.r_at_k_table(data_simple, tp_simple)

"""
def r_at_k_table(data, ground_truth_map):
  r_at_k = []
  header = []
  for path_tup in data:
    vals = get_r_at_k_simple(path_tup[0], ground_truth_map)
    if len(r_at_k) == 0:
      indices = np.arange(len(vals))
      r_at_k.append(indices)
    r_at_k.append(vals)
  return np.array(r_at_k).T

"""
"""
def rank_table(base_dir, gt_map):
  import numpy as np
  n_queries = len(gt_map)
  ranks = np.zeros(n_queries)
  for i in range(0, n_queries):
    filename = base_dir + 'q{:03d}'.format(i) + '_energy_values.csv'
    if not os.path.isfile(filename):
      continue
    energies = np.genfromtxt(filename, delimiter=',', skip_header=1)
    sort_ix = np.argsort(energies[:,1])
    for k in range(0, len(energies)):
      #if do_holdout and energies[sort_ix][k][0] == gt_map[i][0]:
      #  recall[k] = 0.0
      #  continue
      indices = np.array(energies[sort_ix][0:k+1][:,0])
      true_positives = set(indices) & set(gt_map[i])
      if len(true_positives) > 0:
        ranks[i] = k
        break
  return ranks



def find_missing_csv(base_dir):
  import numpy as np
  for i in range(0, 499):
    filename = base_dir + 'q{:03d}_energy_values.csv'.format(i)
    if not os.path.isfile(filename):
      print 'missing {}'.format(filename)



def get_relationship_models(binary_model_mat):
  """Convert the mat file binary model storage to a more convienent structure for python
  Input:
    mat bin model file from sio
      keys
      values
      gmm_params
      platt_params
  Output:
    map from string to relationship_parameters
      'man' -> rel_params
  """
# create a map from trip_string -> index (e.g. 'shirt_on_man' -> 23)
  trip_ix_root = binary_model_mat['binary_models_struct'].s_triple_str_to_idx.serialization
  trip_to_index_keys = trip_ix_root.keys
  trip_to_index_vals = trip_ix_root.values
  trip_str_dict = dict(zip(trip_to_index_keys, trip_to_index_vals))

# for each trip_str key, pull params from the mat and generate a RelationshipParameters object
  param_list = []
  for trip_str in trip_to_index_keys:
    ix = trip_str_dict[trip_str]
    ix -= 1 # MATLAB uses 1-based indexing here
    platt_params = binary_model_mat['binary_models_struct'].platt_models[ix]
    gmm_params = binary_model_mat['binary_models_struct'].models[ix].gmm_params
    rel_params = RelationshipParameters(platt_params[0], platt_params[1], gmm_params.ComponentProportion, gmm_params.mu, gmm_params.Sigma.T)
    param_list.append(rel_params)

  str_to_param_map = dict(zip(trip_to_index_keys, param_list))
  return str_to_param_map



def get_object_detections(image_ix, potentials_mat, platt_mod):
  """Get object detection data from an image
  Input:
    image_ix: image number
    potentials_mat: potentials .mat file
    platt_mod_mat: platt model .mat file
  Output:
    dict: object name (str) -> boxes (numpy array of [x,y,w,h,p] entries), platt model applied to probabilites
  """
  object_mask = [name[:3] == 'obj' for name in potentials_mat['potentials_s'].classes]
  object_mask = np.array(object_mask)
  object_names = potentials_mat['potentials_s'].classes[object_mask]
  object_detections = get_class_detections(image_ix, potentials_mat, platt_mod, object_names)
  return object_detections



def get_attribute_detections(image_ix, potentials_mat, platt_mod):
  """Get object detection data from an image
  Input:
    image_ix: image number
    potentials_mat: potentials .mat file
    platt_mod_mat: platt model .mat file
  Output:
    dict: attribute name (str) -> boxes (numpy array of [x,y,w,h,p] entries), platt model applied to probabilites
  """
  attr_mask = [name[:3] == 'atr' for name in potentials_mat['potentials_s'].classes]
  attr_mask = np.array(attr_mask)
  attr_names = potentials_mat['potentials_s'].classes[attr_mask]
  attr_detections = get_class_detections(image_ix, potentials_mat, platt_mod, attr_names)
  return attr_detections



def get_class_detections(image_ix, potential_data, platt_mod, object_names, verbose=False):
  """Generate box & score values for an image and set of object names
  
  Args:
    image_ix (int): the image to generate detections from
    potential_data (.mat data): potential data (holds boxes, scores, and class to index map)
    platt_data (.mat data): holds platt model parameters
    object_names (numpy array of str): the names of the objects to detect
    verbose (bool): default 'False'
  
  Returns:
    dict: object name (str) -> boxes (numpy array)
  """
  n_objects = object_names.shape[0]
  detections = np.empty(n_objects, dtype=np.ndarray)
  
  box_coords = np.copy(potential_data['potentials_s'].boxes[image_ix])
  box_coords[:,2] -= box_coords[:,0]
  box_coords[:,3] -= box_coords[:,1]
  
  class_to_index_keys = potential_data['potentials_s'].class_to_idx.serialization.keys
  class_to_index_vals = potential_data['potentials_s'].class_to_idx.serialization.values
  obj_id_dict = dict(zip(class_to_index_keys, class_to_index_vals))
  
  det_ix = 0
  for o in object_names:
    if o not in obj_id_dict:
      continue
    
    obj_ix = obj_id_dict[o]
    obj_ix -= 1 # matlab is 1-based
    
    a = 1.0
    b = 1.0
    platt_keys = platt_mod['platt_models'].s_models.serialization.keys
    platt_vals = platt_mod['platt_models'].s_models.serialization.values
    platt_dict = dict(zip(platt_keys, platt_vals))
    if o in platt_dict:
      platt_coeff = platt_dict[o]
      a = platt_coeff[0]
      b = platt_coeff[1]
    
    scores = potential_data['potentials_s'].scores[image_ix][:,obj_ix]
    scores = 1.0 / (1.0 + np.exp(a * scores + b))
    
    n_detections = scores.shape[0]
    scores = scores.reshape(n_detections, 1)
    
    class_det = np.concatenate((box_coords, scores), axis=1)
    detections[det_ix] = class_det
    if verbose: print "%d: %s" % (det_ix, o)
    det_ix += 1
  return dict(zip(object_names, detections))



def get_object_attributes(query):
  """ generate a list of the attributes associated with the objects of a query
    
  The unary_triples field has a subject and object:
  blue    sky   above   green   tree   (query ix 109)
  U1,S0   O1    B1      U2,S1   O2
  subject - index of the object to which the attribute applies (1)
  object - the attribute name (blue)
  attributes 1 1 : 'blue'
             2 1 : 'green'

  Args:
    query (.mat file): an entry in the list of queries from a .mat file

  Returns:
    numpy array: [(0, u'blue'), (1, u'green')]
  """
  n_objects = query.objects.shape[0]
  attributes = []

  if not isinstance(query.unary_triples, np.ndarray):
    node = query.unary_triples
    tup = (node.subject, node.object)
    attributes.append(tup)
  else:
    n_attributes = query.unary_triples.shape[0]
    for attr_ix in range(0, n_attributes):
      node = query.unary_triples[attr_ix]
      tup = (node.subject, node.object)
      attributes.append(tup)
  
  return np.array(attributes)



def get_object_attributes_str(query):
  """ generate a list of the attributes associated with the objects of a query
  Returns:
    list of tuples: [(u'sky', u'blue'), (u'grass', u'green')]
  """
  n_objects = query.objects.shape[0]
  attributes = []
  
  if not isinstance(query.unary_triples, np.ndarray):
    node = query.unary_triples
    sub_ix = node.subject
    sub_name = query.objects[sub_ix].names
    if isinstance(sub_name, np.ndarray):
      sub_name = sub_name[0]
    tup = (sub_name, node.object)
    attributes.append(tup)
  else:
    n_attributes = query.unary_triples.shape[0]
    for attr_ix in range(0, n_attributes):
      node = query.unary_triples[attr_ix]
      sub_ix = node.subject
      sub_name = query.objects[sub_ix].names
      if isinstance(sub_name, np.ndarray):
        sub_name = sub_name[0]
      tup = (sub_name, node.object)
      attributes.append(tup)
  
  return attributes



def get_partial_scene_matches(images, scenes):
  matches = []
  for q_ix in range(0, len(scenes)):
    scene = scenes[q_ix].annotations
    matching_images = []
    for i_ix in range(0, len(images)):
      image = images[i_ix].annotations
      if does_match(image, scene):
        matching_images.append(i_ix)
    matches.append(matching_images)
  return np.array(matches)



# 519/635 & 0 is a good match (clear glasses on woman)
def does_match(image, scene):
  verbose = False
  
  # are all scene objects in the image?
  scene_objects = []
  for scene_obj_ix in range(0, len(scene.objects)):
    scene_obj_name = scene.objects[scene_obj_ix].names
    scene_objects.append(scene_obj_name)
  
  image_objects = []
  for image_obj_ix in range(0, len(image.objects)):
    image_obj_name = image.objects[image_obj_ix].names
    if isinstance(image_obj_name, np.ndarray):
      image_obj_name = image_obj_name[0]
    image_objects.append(image_obj_name)
  
  is_subset = set(scene_objects).issubset(set(image_objects))
  if not is_subset:
    if verbose: print '{} not ss of {}'.format(scene_objects, image_objects)
    return False
  
  # are all scene object attributes in the image?
  scene_oa = get_object_attributes_str(scene)
  image_oa = get_object_attributes_str(image)
  is_subset = set(scene_oa).issubset(set(image_oa))
  if not is_subset:
    if verbose: print '{} not ss of {}'.format(scene_oa, image_oa)
    return False

  # are all scene relationships in the image?
  scene_triples = []
  if isinstance(scene.binary_triples, np.ndarray):
    for trip in scene.binary_triples:
      scene_triples.append(trip)
  else:
    scene_triples.append(scene.binary_triples)
  
  image_triples = []
  if isinstance(image.binary_triples, np.ndarray):
    for trip in image.binary_triples:
      image_triples.append(trip)
  else:
    image_triples.append(image.binary_triples)
  
  scene_rels = []
  for scene_trip in scene_triples:
    scene_sub_ix = scene_trip.subject
    scene_sub_name = scene.objects[scene_sub_ix].names
    
    scene_obj_ix = scene_trip.object
    scene_obj_name = scene.objects[scene_obj_ix].names
    
    rel_str = '{} {} {}'.format(scene_sub_name, scene_trip.predicate, scene_obj_name)
    scene_rels.append(rel_str)
  
  image_rels = []
  for image_trip in image_triples:
    rel_str = '{} {} {}'.format(image_trip.text[0], image_trip.text[1], image_trip.text[2])
    image_rels.append(rel_str)
  
  is_subset = set(scene_rels).issubset(set(image_rels))
  if not is_subset:
    if verbose: print '{} not ss of {}'.format(scene_rels, image_rels)
  return is_subset



def get_rel_string(relationship, objects):
  sub_ix = relationship.subject
  sub_name = objects[sub_ix].names
  if isinstance(sub_name, np.ndarray):
    sub_name = sub_name[0]
  obj_ix = relationship.object
  obj_name = objects[obj_ix].names
  if isinstance(obj_name, np.ndarray):
    obj_name = obj_name[0]
  return '{} {} {}'.format(sub_name, predicate, object)



def obscure(query):
  """ obscure the name of the first object with an attribute in the query
  Deep copies the original and returns the obscured query
  Ret:
    sio.mat_struct : the altered query
  """
  import copy as copy
  oa_list = get_object_attributes(query)
  q_copy = copy.deepcopy(query)
  obj_ix = 0
  if len(oa_list) > 0:
    obj_ix = int(oa_list[0][0])
  q_copy.objects[obj_ix].names = '_' + q_copy.objects[obj_ix].names + '_'
  return q_copy



#===============================================================================
# GENERAL UTILITIES
#===============================================================================

def sg_to_str(scene):
  attrs = get_object_attributes(scene)
  
  # get relationship string
  pred_name = scene.binary_triples.predicate
  
  # get the subject index and name
  subject_ix = scene.binary_triples.subject
  subject_name = scene.objects[subject_ix].names
  
  # get the object index and name
  object_ix = scene.binary_triples.object
  object_name = scene.objects[object_ix].names
  
  #get the attributes
  sub_attrs = []
  sub_attr_str = ''
  
  obj_attrs = []
  obj_attr_str = ''
  
  for attr in attrs:
    if int(attr[0]) == subject_ix:
      sub_attrs.append(attr[1])
      sub_attr_str = sub_attr_str + attr[1] + ' '
    elif int(attr[0]) == object_ix:
      obj_attrs.append(attr[1])
      obj_attr_str = obj_attr_str + attr[1] + ' '
  
  # generate attribute strings
  
  # format the string
  ret_str = '{}{} {} {}{}'.format(sub_attr_str, subject_name, pred_name, obj_attr_str, object_name)
  return ret_str



#-------------------------------------------------------------------------------
# Get general metrics about a factor model
def detail_pgm(gm):
  import opengm as ogm
  
  bin_avg_sum = 0.
  n_bin = 0
  unary_avg_sum = 0.
  n_unary = 0
  
  print 'ix, fn_type,   min ,   max ,   avg , median' 
  for factor, factor_ix in gm.factorsAndIds():
    func = np.array(factor)
    #func = np.exp(-func)
    f_min = np.min(func)
    f_max = np.max(func)
    f_avg = np.average(func)
    f_mdn = np.median(func)
    
    f_type = ''
    if len(func.shape) == 2:
      f_type = 'binary'
      bin_avg_sum += f_avg
      n_bin += 1
    else:
      f_type = 'unary'
      unary_avg_sum += f_avg
      n_unary += 1
    
    print '{:02d}, {:>7}, {:6.2f}, {:6.2f}, {:6.2f}, {:6.2f}'.format(factor_ix, f_type, f_min, f_max, f_avg, f_mdn)
  
  # all unaries/binaries have the same denominator when averaged so this is ok
  unary_avg = unary_avg_sum / n_unary
  binary_avg = bin_avg_sum / n_bin
  print 'unary func avg :   ll: {:6.2f}   lik: {:6.2f}'.format(unary_avg, np.exp(unary_avg))
  print 'binary func avg:   ll: {:6.2f}   lik: {:6.2f}'.format(binary_avg, np.exp(binary_avg))



#-------------------------------------------------------------------------------
# Get general metrics about an image set's ground truth queries
#-------------------------------------------------------------------------------
def detail_image_set(vg_data):
  stats = []
  for i in range(0, len(vg_data['vg_data_test'])):
    img_root = vg_data['vg_data_test'][i].annotations
    n_objects = len(img_root.objects)
    n_attributes = len(img_root.unary_triples)
    n_relations = len(img_root.binary_triples)
    stats.append((i, n_objects, np.round(n_relations / (n_objects * 1.), decimals=2), n_attributes, n_relations))
  return np.array(stats, dtype=object)



#-------------------------------------------------------------------------------
# print the objects, attributes, and relationships in a dataset
#-------------------------------------------------------------------------------
# printImageDetails(vgd['vg_data_test'], 0)
#-------------------------------------------------------------------------------
def printImageDetails(data_set, image_num):
  img_root = data_set[image_num].annotations
  print "IMAGE {}:".format(image_num)
  print("OBJECTS:")
  print("--------------------------------------------------------------------------------")
  for i in range(0, img_root.objects.shape[0]):
    o = img_root.objects[i]
    print "%d: %s at (%d, %d, %d, %d)" % (i, o.names, o.bbox.x, o.bbox.y, o.bbox.w, o.bbox.h)

  print("\r\nATTRIBUTES:")
  print("--------------------------------------------------------------------------------")
  for i in range(0, img_root.unary_triples.shape[0]):
    o = img_root.unary_triples[i]
    print "%i: %s | %i" % (i, o.text, o.subject)

  print("\r\nRELATIONSHIPS:")
  print("--------------------------------------------------------------------------------")
  for i in range(0, img_root.binary_triples.shape[0]):
    print "%i: %s | %i -> %i" % (i, img_root.binary_triples[i].text, img_root.binary_triples[i].subject, img_root.binary_triples[i].object)

def print_all_image_details(data_set):
  for i in range(0, len(data_set)):
    printImageDetails(data_set, i)


#-------------------------------------------------------------------------------
# find instances of a relationship in the data set: <sub> <rel> <obj>
#-------------------------------------------------------------------------------
# subject - subject of the relationship
# predicate - predicate of the relationship
# object - the object if the relationship
# data_root - root of the test/training data (e.g. vgd['vd_data_test'])
#
# findRelationship("monitor", "on", "desk", vgd['vg_data_test'])
#-------------------------------------------------------------------------------
def findRelationship(subject_name, predicate, object_name, vgd):
  for img in range(0, vgd['vg_data_test'].shape[0]):
    anno_root = vgd['vg_data_test'][img].annotations

    for i in range(0, anno_root.binary_triples.shape[0]):
      rel = anno_root.binary_triples[i].predicate
      if (rel == predicate):
        sub_ix = anno_root.binary_triples[i].subject
        sub_name = anno_root.objects[sub_ix].names
        if not isinstance(sub_name, unicode):
          sub_name = sub_name[0]

        obj_ix = anno_root.binary_triples[i].object
        obj_name = anno_root.objects[obj_ix].names
        if not isinstance(obj_name, unicode):
          obj_name = obj_name[0]

        if sub_name == subject_name and obj_name == object_name:
          print "img %i: %s" % (img, vgd['vg_data_test'][img].image_url)
          break; # sometimes there are multiple instances



#-------------------------------------------------------------------------------
# Print the queries in a set of simple queries
#-------------------------------------------------------------------------------
# Assumes a set of  simple binary relationship queries
#   e.g. monitor on desk, sign on pole, man wearing hat
#
# getQueries(queries['simple_graphs'])
#-------------------------------------------------------------------------------
def getQueries(q_root):
  for q in range(0, q_root.shape[0]):
    subject_ix = q_root[q].annotations.binary_triples.subject
    subject_name = q_root[q].annotations.objects[subject_ix].names
    
    object_ix = q_root[q].annotations.binary_triples.object
    object_name = q_root[q].annotations.objects[object_ix].names
    
    pred_name = q_root[q].annotations.binary_triples.predicate
    
    attrs = get_object_attributes_str(q_root[q].annotations)
    print "Q%d: %s %s %s (%s)" % (q, subject_name, pred_name, object_name, attrs)



#-------------------------------------------------------------------------------
# Snippet for finding which scene graphs have multiple attributes
#-------------------------------------------------------------------------------
'''
for i in range(0, queries['simple_graphs'].shape[0]):
  trip = queries['simple_graphs'][i].annotations.unary_triples
  if isinstance(trip, np.ndarray):
    for j in range(0, trip.shape[0]):
      print "%d: (%d <- %s)" % (i, trip[j].subject, trip[j].object)
  else:
    print "%d: (%d <- %s)" % (i, trip.subject, trip.object)

for i in range(0, queries['simple_graphs'].shape[0]):
  trip = queries['simple_graphs'][i].annotations.binary_triples
  if isinstance(trip, np.ndarray):
    for j in range(0, trip.shape[0]):
      print "%d: %d %s %d" % (i, trip[j].subject, trip[j].predicate, trip[j].object)
  else:
    print "%d: %d %s %d" % (i, trip.subject, trip.predicate, trip.object)
'''
