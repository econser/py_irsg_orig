import opengm as ogm
import numpy as np
import scipy.io as sio
import itertools
import os.path
from scipy.stats import multivariate_normal as mvn



#===============================================================================
# Locally used classes
#===============================================================================
class FuncDetail(object):
  """Gathers OpenGM function information in a single class
  
  Attributes:
    gm_function_id (int): OpenGM function ID
    function_type (str): description of the function type
    source (str): high-level description of the function
    detail (str): detailed description of the function
  """
  def __init__(self, gm_function_id, var_indices, function_type, source, detail):
    self.gm_function_id = gm_function_id
    self.var_indices = var_indices
    self.function_type = function_type
    self.source = source
    self.detail = detail



class ModelState(object):
  """Utility class for gathering image inspection information together in one place
  
  Attributes:
    object_box (numpy 4,1): x,y,w,h box for the object of the relationship
    subject_box (numpy 4,1): x,y,w,h box for the subject of the relationship
    ll_object_class (float): log likelihood of the object box being of the relationship's object class
    ll_subject_class (float): log likelihood of the subject box being of the relationship's subject class
    ll_relationship (float): log likelihood of the subject-predicate-object relationship
    ll_composite (float): sum of subject, relationship, object values
  """
  def __init__(self, object_box = None, subject_box = None, ll_object_class = 0.0, ll_subject_class = 0.0, ll_relationship = 0.0, ll_composite = 0.0):
    self.object_box = object_box
    self.subject_box = subject_box
    self.ll_object_class = ll_object_class
    self.ll_subject_class = ll_subject_class
    self.ll_relationship = ll_relationship
    self.ll_composite = ll_composite
  
  def __lt__(self, other):
    """less than method for sorting"""
    return self.ll_composite < other.ll_composite



class BinDets (object):
  def __init__(self, description, factor_mtx, object_name, predicate, subject_name):
    self.description = description
    self.factor = factor_mtx
    self.object_name = object_name
    self.predicate_name = predicate_name
    self.subject_name = subject_name

class UnaryDets (object):
  def __init__(self, description, factor_array):
    self.description = description
    self.factor_array = factor_array

class _DetectionTracker (object):
  def __init__(self, image_path):
    self.image_path = image_path
    self.box_coords = None
    self.object_names = []

class DetectionGroup (object):
  def __init__(self, rel_str, ll_relationship, object_boxes, object_name, subject_boxes, subject_name):
    self.relationship = rel_str
    self.ll_relationship = ll_relationship
    self.object_boxes = object_boxes
    self.object_name = object_name
    self.subject_boxes = subject_boxes
    self.subject_name = subject_name
  
class DetectionTracker (object):
  def __init__(self, image_path):
    self.image_path = image_path
    self.box_pairs = None
    self.object_names = []
    self.box_coords = None
    self.unary_detections = []
    self.detected_vars = []
    self.relationships = []
  
  def add_group(self, rel_str, ll_relationship, object_boxes, object_name, subject_boxes, subject_name):
    grp = DetectionGroup(rel_str, ll_relationship, object_boxes, object_name, subject_boxes, subject_name)
    self.relationships.append(grp)



class PyCallback(object):
  def __init__(self):
    visit_count = 0
  def begin(self,inference):
    print "begin"
  def end(self,inference):
    print "end"
  def visit(self,inference):
    arg=inference.arg()
    gm=inference.gm()
    print "  visit {0}: energy {1}".format(visit_count, gm.evaluate(arg))
    visit_count += 1



#===============================================================================
# Generate PGM
# query_graph <- 
# object_detections <- get_object_detections()
# attribute_detections <- get_attribute_detections()
# relationship_models <- get_relationship_models(<binary_model_mat_file>)
# per_object_attributes <- get_object_attributes(<scenegraph_query>)
# image_filename <- <fq_filename>
# verbose <- verbosity flag (True/False)
#===============================================================================
from line_profiler import LineProfiler
def do_profile(follow=[]):
  def inner(func):
    def profiled_func(*args, **kwargs):
      try:
        profiler = LineProfiler()
        profiler.add_function(func)
        for f in follow:
          profiler.add_function(f)
        profiler.enable_by_count()
        return func(*args, **kwargs)
      finally:
        profiler.print_stats()
    return profiled_func
  return inner
#@do_profile()
def generate_pgm(if_data, verbose=False):
  # gather data from the if data object
  query_graph = if_data.current_sg_query
  object_detections = if_data.object_detections
  attribute_detections = if_data.attribute_detections
  relationship_models = if_data.relationship_models
  per_object_attributes = if_data.per_object_attributes
  image_filename = if_data.image_filename
  
  # generate the graphical model (vg_data_build_gm_for_image)
  n_objects = len(query_graph.objects)
  n_vars = []
  object_is_detected = []
  query_to_pgm = []
  pgm_to_query = []

  master_box_coords = []
  
  varcount = 0
  for obj_ix in range(0, n_objects):
    query_object_name = query_graph.objects[obj_ix].names
    
    # occasionally, there are multiple object names (is 0 the best?)
    if isinstance(query_object_name, np.ndarray):
      query_object_name = query_object_name[0]
    
    object_name = "obj:" + query_object_name
    if object_name not in object_detections:
      object_is_detected.append(False)
      query_to_pgm.append(-1)
    else:
      if len(master_box_coords) == 0:
        master_box_coords = np.copy(object_detections[object_name][:,0:4])
      object_is_detected.append(True)
      query_to_pgm.append(varcount)
      varcount += 1
      pgm_to_query.append(obj_ix)
      
      n_labels = len(object_detections[object_name])
      n_vars.append(n_labels)
  
  gm = ogm.gm(n_vars, operator='adder')
  if verbose:
    print "number of variables: {0}".format(gm.numberOfVariables)
    for l in range(0, gm.numberOfVariables):
      print "  labels for var {0}: {1}".format(l, gm.numberOfLabels(l))
  
  functions = []
  
  # generate 1st order functions for objects
  # TODO: test an uniform dist for missing objects
  if verbose: print "unary functions - objects:"
  
  unary_dets = []
  is_cnn_detected = []
  for obj_ix in range(0, n_objects):
    fid = None
    
    pgm_ix = query_to_pgm[obj_ix]
    object_name = query_graph.objects[obj_ix].names
    if isinstance(object_name, np.ndarray):
      object_name = object_name[0]
    detail = "unary function for object '{0}'".format(object_name)
    
    if object_is_detected[obj_ix]:
      if verbose: print "  adding {0} as full explicit function (qry_ix:{1}, pgm_ix:{2})".format(detail, obj_ix, pgm_ix)
      is_cnn_detected.append(True)
      prefix_object_name = "obj:" + object_name
      detections = object_detections[prefix_object_name]
      unary_dets.append(detections[:,4])
      log_scores = -np.log(detections[:,4])
      fid = gm.addFunction(log_scores)
    else:
      if verbose: print "  skipping {0}, no detection available (qry_ix:{1})".format(object_name, obj_ix)
      continue
    
    func_detail = FuncDetail(fid, [pgm_ix], "explicit", "object unaries", detail)
    functions.append(func_detail)
  
# generate 1st order functions for attributes
  if verbose: print "unary functions - attributes:"
  n_attributes = len(per_object_attributes)
  for attr_ix in range(0, n_attributes):
    obj_ix = int(per_object_attributes[attr_ix][0])
    pgm_ix = query_to_pgm[obj_ix]
    attribute_name = per_object_attributes[attr_ix][1]
    prefix_attribute_name = "atr:" + attribute_name
    
    if prefix_attribute_name not in attribute_detections:
      if verbose: print "  skipping attribute '{0}' for object '{1}' (qry_ix:{2}), no attribute detection available".format(attribute_name, query_graph.objects[obj_ix].names, obj_ix)
      continue
    
    if not object_is_detected[obj_ix]:
      if verbose: print "  skipping attribute '{0}' for object '{1}' (qry_ix:{2}), no object detection available".format(attribute_name, query_graph.objects[obj_ix].names, obj_ix)
      continue
    
    detections = attribute_detections[prefix_attribute_name]
    log_scores = -np.log(detections[:,4])
    
    detail = "unary function for attribute '{0}' of object '{1}' (qry_ix:{2}, pgm_ix:{3})".format(attribute_name, query_graph.objects[obj_ix].names, obj_ix, pgm_ix)
    if verbose: print "  adding {0}".format(detail)
    
    fid = gm.addFunction(log_scores)
    func_detail = FuncDetail(fid, [pgm_ix], "explicit", "attribute unaries", detail)
    functions.append(func_detail)
    
  # generate a tracker for storing obj/attr/rel likelihoods (pre-inference)
  tracker = DetectionTracker(image_filename)
  for i in range(0, n_objects):
    if object_is_detected[i]:
      if isinstance(query_graph.objects[i].names, np.ndarray):
        tracker.object_names.append(query_graph.objects[i].names[0])
      else:
        tracker.object_names.append(query_graph.objects[i].names)
  tracker.unary_detections = unary_dets
  tracker.box_coords = master_box_coords
  tracker.detected_vars = is_cnn_detected
  
  # generate 2nd order functions for binary relationships
  trip_root = query_graph.binary_triples
  trip_list = []
  if isinstance(trip_root, sio.matlab.mio5_params.mat_struct):
    trip_list.append(query_graph.binary_triples)
  else:
    # if there's only one relationship, we don't have an array :/
    for trip in trip_root:
      trip_list.append(trip)
  
  # generate a single cartesian product of the boxes
  # this will only work when all objects are detected across the same boxes
  # we know this is the case for this implementation
  master_cart_prod = None
  for i in range(0, n_objects):
    if object_is_detected[i]:
      obj_name = query_graph.objects[i].names
      boxes = None
      if isinstance(obj_name, np.ndarray):
        boxes = object_detections["obj:"+obj_name[0]]
      else:
        boxes = object_detections["obj:"+obj_name]
      master_cart_prod = np.array([x for x in itertools.product(boxes, boxes)])
      break
  tracker.box_pairs = master_cart_prod

  # process each binary triple in the list
  if verbose: print "binary functions:"
  for trip in trip_list:
    sub_ix = trip.subject
    sub_pgm_ix = query_to_pgm[sub_ix]
    subject_name = query_graph.objects[sub_ix].names
    if isinstance(subject_name, np.ndarray):
      subject_name = subject_name[0]
    
    obj_ix = trip.object
    obj_pgm_ix = query_to_pgm[obj_ix]
    object_name = query_graph.objects[obj_ix].names
    if isinstance(object_name, np.ndarray):
      object_name = object_name[0]
    
    relationship = trip.predicate
    bin_trip_key = subject_name + "_" + relationship.replace(" ", "_")  + "_" + object_name
   
    # check if there is a gmm for the specific triple string
    if bin_trip_key not in relationship_models:
      if verbose: print "  no model for '{0}', generating generic relationship string".format(bin_trip_key)
      bin_trip_key = "*_" + relationship.replace(" ", "_") + "_*"
      if bin_trip_key not in relationship_models:
        if verbose: print "    skipping binary function for relationship '{0}', no model available".format(bin_trip_key)
        continue
    
    # verify object detections
    if sub_ix == obj_ix:
      if verbose: print "    self-relationships not possible in OpenGM, skipping relationship"
      continue
    
    if not object_is_detected[sub_ix]:
      if verbose: print "    no detections for object '{0}', skipping relationship (qry_ix:{1})".format(subject_name, sub_ix)
      continue
    
    if not object_is_detected[obj_ix]:
      if verbose: print "    no detections for object '{0}', skipping relationship (qry_ix:{1})".format(object_name, obj_ix)
      continue
    
    # get model parameters
    prefix_object_name = "obj:" + object_name
    bin_object_box = object_detections[prefix_object_name]
    
    prefix_subject_name = "obj:" + subject_name
    bin_subject_box = object_detections[prefix_subject_name]
    
    rel_params = relationship_models[bin_trip_key]
    
    # generate features from subject and object detection boxes
    cart_prod = master_cart_prod
    sub_dim = 0
    obj_dim = 1
    
    subx_center = cart_prod[:, sub_dim, 0] + 0.5 * cart_prod[:, sub_dim, 2]
    suby_center = cart_prod[:, sub_dim, 1] + 0.5 * cart_prod[:, sub_dim, 3]
    
    objx_center = cart_prod[:, obj_dim, 0] + 0.5 * cart_prod[:, obj_dim, 2]
    objy_center = cart_prod[:, obj_dim, 1] + 0.5 * cart_prod[:, obj_dim, 3]
    
    sub_width = cart_prod[:, sub_dim, 2]
    relx_center = (subx_center - objx_center) / sub_width
    
    sub_height = cart_prod[:, sub_dim, 3]
    rely_center = (suby_center - objy_center) / sub_height
    
    rel_height = cart_prod[:, obj_dim, 2] / cart_prod[:, sub_dim, 2]
    rel_width = cart_prod[:, obj_dim, 3] / cart_prod[:, sub_dim, 3]
    
    features = np.vstack((relx_center, rely_center, rel_height, rel_width)).T
    
    #tracker.box_pairs = np.copy(cart_prod) #TODO: is this copy necessary?
    #tracker.box_pairs = cart_prod
    
    # generate scores => log(epsilon+scores) => platt sigmoid
    scores = gmm_pdf(features, rel_params.gmm_weights, rel_params.gmm_mu, rel_params.gmm_sigma)
    eps = np.finfo(np.float).eps
    scores = np.log(eps + scores)
    sig_scores = 1.0 / (1. + np.exp(rel_params.platt_a * scores + rel_params.platt_b))
    
    log_likelihoods = -np.log(sig_scores)
    
    #tracker.add_group(bin_trip_key, np.copy(log_likelihoods), np.copy(bin_object_box), object_name, np.copy(bin_subject_box), subject_name) # TODO: are these copy calls necessary?
    tracker.add_group(bin_trip_key, log_likelihoods, bin_object_box, object_name, bin_subject_box, subject_name)
    
    # generate the matrix of functions
    n_subject_val = len(bin_subject_box)
    n_object_val = len(bin_object_box)
    bin_functions = np.reshape(log_likelihoods, (n_subject_val, n_object_val)) # TODO: determine if any transpose is needed
    if obj_pgm_ix < sub_pgm_ix: bin_functions = bin_functions.T
    
    # add binary functions to the GM
    detail = "binary functions for relationship '%s'" % (bin_trip_key)
    if verbose: print("    adding %s" % detail)
    fid = gm.addFunction(bin_functions)
    
    var_indices = [sub_pgm_ix, obj_pgm_ix]
    if obj_pgm_ix < sub_pgm_ix: var_indices = [obj_pgm_ix, sub_pgm_ix]
    func_detail = FuncDetail(fid, var_indices, "explicit", "binary functions", detail)
    functions.append(func_detail)
    
  # add 1st order factors (fid)
  for f in functions:
    n_indices = len(f.var_indices)
    if n_indices == 1:
      if verbose:
        print "  adding unary factor: {0}".format(f.detail)
        print "    fid: {0}   var: {1}".format(f.gm_function_id.getFunctionIndex(), f.var_indices[0])
      gm.addFactor(f.gm_function_id, f.var_indices[0])
    elif n_indices == 2:
      if verbose:
        print "  adding binary factor: {0}".format(f.detail)
        print "    fid: {0}   var: {1}".format(f.gm_function_id.getFunctionIndex(), f.var_indices)
      gm.addFactor(f.gm_function_id, f.var_indices)
    else:
      if verbose: print "skipping unexpected factor with {0} indices: {1}".format(n_indices, f.function_type)
  
  return gm, tracker



#================================================================================
#  Generate a PGM with uniform vars for missing objects
#================================================================================
def generate_pgm_all_objects(if_data, method='uniform', verbose=False):
  """ Generate Uniform Detection Subustution PGM
  Substitutes a uniform distribution for objects with no detections
  
  Attributes:
    if_data: a configured ImageFetchDataset object
    method (string): when objects do not have RCNN detections, replace with:
      'uniform' - a uniform distribution p(x) = 1. / number_of_boxes
      'empirical' - each box takes the average of all detected objects for this image
  """
  # gather data from the if data object
  query_graph = if_data.current_sg_query
  object_detections = if_data.object_detections
  attribute_detections = if_data.attribute_detections
  relationship_models = if_data.relationship_models
  per_object_attributes = if_data.per_object_attributes
  image_filename = if_data.image_filename
  
  # prep vars that we'll need for construction of the GM
  n_objects = len(query_graph.objects)
  n_vars = []
  object_is_detected = []
  query_to_pgm = []
  pgm_to_query = []
  
  master_box_coords = []
  master_empirical_dist = None

  # determine the number of variables and labels
  varcount = 0
  for obj_ix in range(0, n_objects):
    query_object_name = query_graph.objects[obj_ix].names
    
    # occasionally, there are multiple object names (is 0 the best?)
    if isinstance(query_object_name, np.ndarray):
      query_object_name = query_object_name[0]
    
    object_name = "obj:" + query_object_name
    if object_name not in object_detections:
      object_is_detected.append(False)
    else:
      object_is_detected.append(True)
      # generate the empirical distribution while we're checking for detections
      if master_empirical_dist is None:
        master_empirical_dist = np.copy(object_detections[object_name][:,4])
      else:
        master_empirical_dist += object_detections[object_name][:,4]
    
    query_to_pgm.append(varcount)
    pgm_to_query.append(obj_ix)
    varcount += 1
    
    # make a copy of the box coordinates for use in the empirical/uniform dists
    n_labels = 0
    if object_is_detected[obj_ix]:
      n_labels = len(object_detections[object_name])
      if len(master_box_coords) == 0:
        master_box_coords = np.copy(object_detections[object_name][:,0:4])
    n_vars.append(n_labels)
  
  # update the number of labels for missing detections
  labels_per_var = max(n_vars)
  for label_ix in range(0, len(n_vars)):
    if n_vars[label_ix] == 0:
      n_vars[label_ix] = labels_per_var
  
  # generate master uniform/implicit functions
  master_unif_likelihoods = np.ones(labels_per_var) * 1./labels_per_var
  master_unif_likelihoods = np.reshape(master_unif_likelihoods, (labels_per_var, 1))
  
  master_empirical_dist *= 1. / np.sum(object_is_detected)
  master_empirical_dist = np.reshape(master_empirical_dist, (labels_per_var, 1))

  master_unif_detections = None
  if method == 'uniform':
    master_unif_detections = np.hstack((master_box_coords, master_unif_likelihoods))
  elif method == 'empirical':
    master_unif_detections = np.hstack((master_box_coords, master_empirical_dist))
  else:
    raise ValueError('unexpected method "{}", expected "uniform" or "empirical"'.format(method))
  
  # now that we know the numer of vars and labels, instantiate the GM
  gm = ogm.gm(n_vars, operator='adder')
  if verbose:
    print "number of variables: {0}".format(gm.numberOfVariables)
    for l in range(0, gm.numberOfVariables):
      print "  labels for var {0}: {1}".format(l, gm.numberOfLabels(l))
  
  # keep track of all of the functions for the GM, we'll add them to the GM at the end
  functions = []
  
  # generate 1st order functions (detections) for objects
  if verbose: print "unary functions - objects:"
  
  unary_dets = []
  for obj_ix in range(0, n_objects):
    fid = None
    detections = None
    
    pgm_ix = query_to_pgm[obj_ix]
    object_name = query_graph.objects[obj_ix].names
    if isinstance(object_name, np.ndarray):
      object_name = object_name[0]
    detail = "unary function for object '{0}'".format(object_name)
    
    if object_is_detected[obj_ix]:
      if verbose: print "  adding {0} as full explicit function (qry_ix:{1}, pgm_ix:{2})".format(detail, obj_ix, pgm_ix)
      prefix_object_name = "obj:" + object_name
      detections = object_detections[prefix_object_name]
    else:
      if verbose: print "  adding uniform detections for {0}, no detection available (qry_ix:{1})".format(object_name, obj_ix)
      detections = master_unif_detections
    
    unary_dets.append(detections[:,4])
    log_scores = -np.log(detections[:,4])
    fid = gm.addFunction(log_scores)
    func_detail = FuncDetail(fid, [pgm_ix], "explicit", "object unaries", detail)
    functions.append(func_detail)
  
  # generate 1st order functions (detections) for attributes
  if verbose: print "unary functions - attributes:"
  n_attributes = len(per_object_attributes)
  for attr_ix in range(0, n_attributes):
    obj_ix = int(per_object_attributes[attr_ix][0])
    pgm_ix = query_to_pgm[obj_ix]
    attribute_name = per_object_attributes[attr_ix][1]
    prefix_attribute_name = "atr:" + attribute_name
    
    # skip attrs with no detections rather than try to invent a detection
    if prefix_attribute_name not in attribute_detections:
      if verbose: print "  skipping attribute '{0}' for object '{1}' (qry_ix:{2}), no attribute detection available".format(attribute_name, query_graph.objects[obj_ix].names, obj_ix)
      continue
    
    detections = attribute_detections[prefix_attribute_name]
    log_scores = -np.log(detections[:,4])
    
    detail = "unary function for attribute '{0}' of object '{1}' (qry_ix:{2}, pgm_ix:{3})".format(attribute_name, query_graph.objects[obj_ix].names, obj_ix, pgm_ix)
    if verbose: print "  adding {0}".format(detail)
    
    fid = gm.addFunction(log_scores)
    func_detail = FuncDetail(fid, [pgm_ix], "explicit", "attribute unaries", detail)
    functions.append(func_detail)
    
  # generate a tracker for storing obj/attr/rel likelihoods (pre-inference)
  tracker = DetectionTracker(image_filename)
  for i in range(0, n_objects):
    if isinstance(query_graph.objects[i].names, np.ndarray):
      tracker.object_names.append(query_graph.objects[i].names[0])
    else:
      tracker.object_names.append(query_graph.objects[i].names)
  tracker.unary_detections = unary_dets
  tracker.box_coords = master_box_coords
  tracker.detected_vars = object_is_detected
  
  # generate 2nd order functions for binary relationships
  trip_root = query_graph.binary_triples
  trip_list = []
  if isinstance(trip_root, sio.matlab.mio5_params.mat_struct):
    trip_list.append(query_graph.binary_triples)
  else:
    # if there's only one relationship, we don't have an array :/
    for trip in trip_root:
      trip_list.append(trip)
  
  # generate a single cartesian product of the boxes
  # this will only work when all objects are detected across the same boxes
  # we know this is the case for this implementation
  cart_prod_iter = itertools.product(master_box_coords, master_box_coords)
  master_cart_prod = np.array([x for x in cart_prod_iter])
  tracker.box_pairs = master_cart_prod
  
  # process each binary triple in the list
  if verbose: print "binary functions:"
  for trip in trip_list:
    sub_ix = trip.subject
    sub_pgm_ix = query_to_pgm[sub_ix]
    subject_name = query_graph.objects[sub_ix].names
    if isinstance(subject_name, np.ndarray):
      subject_name = subject_name[0]
    
    obj_ix = trip.object
    obj_pgm_ix = query_to_pgm[obj_ix]
    object_name = query_graph.objects[obj_ix].names
    if isinstance(object_name, np.ndarray):
      object_name = object_name[0]
    
    relationship = trip.predicate
    bin_trip_key = subject_name + "_" + relationship.replace(" ", "_")  + "_" + object_name
   
    # check if there is a gmm for the specific triple string
    if bin_trip_key not in relationship_models:
      if verbose: print "  no model for '{0}', generating generic relationship string".format(bin_trip_key)
      bin_trip_key = "*_" + relationship.replace(" ", "_") + "_*"
      if bin_trip_key not in relationship_models:
        if verbose: print "    skipping binary function for relationship '{0}', no model available".format(bin_trip_key)
        continue
    
    # verify object detections
    if sub_ix == obj_ix:
      if verbose: print "    self-relationships not possible in OpenGM, skipping relationship"
      continue
    
    # get model parameters
    bin_object_box = None
    if object_is_detected[obj_ix]:
      prefix_object_name = "obj:" + object_name
      bin_object_box = object_detections[prefix_object_name]
    else:
      bin_object_box = np.copy(master_unif_detections)
    
    bin_subject_box = None
    if object_is_detected[sub_ix]:
      prefix_subject_name = "obj:" + subject_name
      bin_subject_box = object_detections[prefix_subject_name]
    else:
      bin_subject_box = np.copy(master_unif_detections)
    
    rel_params = relationship_models[bin_trip_key]
    
    # generate features from subject and object detection boxes
    cart_prod = master_cart_prod
    sub_dim = 0
    obj_dim = 1
    
    subx_center = cart_prod[:, sub_dim, 0] + 0.5 * cart_prod[:, sub_dim, 2]
    suby_center = cart_prod[:, sub_dim, 1] + 0.5 * cart_prod[:, sub_dim, 3]
    
    objx_center = cart_prod[:, obj_dim, 0] + 0.5 * cart_prod[:, obj_dim, 2]
    objy_center = cart_prod[:, obj_dim, 1] + 0.5 * cart_prod[:, obj_dim, 3]
    
    sub_width = cart_prod[:, sub_dim, 2]
    relx_center = (subx_center - objx_center) / sub_width
    
    sub_height = cart_prod[:, sub_dim, 3]
    rely_center = (suby_center - objy_center) / sub_height
    
    rel_height = cart_prod[:, obj_dim, 2] / cart_prod[:, sub_dim, 2]
    rel_width = cart_prod[:, obj_dim, 3] / cart_prod[:, sub_dim, 3]
    
    features = np.vstack((relx_center, rely_center, rel_height, rel_width)).T
    
    # generate scores => log(epsilon+scores) => platt sigmoid
    scores = gmm_pdf(features, rel_params.gmm_weights, rel_params.gmm_mu, rel_params.gmm_sigma)
    eps = np.finfo(np.float).eps
    log_scores = np.log(eps + scores)
    sig_scores = 1.0 / (1. + np.exp(rel_params.platt_a * log_scores + rel_params.platt_b))
    
    log_likelihoods = -np.log(sig_scores)
    
    tracker.add_group(bin_trip_key, log_likelihoods, bin_object_box, object_name, bin_subject_box, subject_name)
    
    # generate the matrix of functions
    n_subject_val = len(bin_subject_box)
    n_object_val = len(bin_object_box)
    bin_functions = np.reshape(log_likelihoods, (n_subject_val, n_object_val)) # TODO: determine if any transpose is needed
    if obj_pgm_ix < sub_pgm_ix: bin_functions = bin_functions.T
    
    # add binary functions to the GM
    detail = "binary functions for relationship '%s'" % (bin_trip_key)
    if verbose: print '    adding {}'.format(detail)
    fid = gm.addFunction(bin_functions)
    
    var_indices = [sub_pgm_ix, obj_pgm_ix]
    if obj_pgm_ix < sub_pgm_ix: var_indices = [obj_pgm_ix, sub_pgm_ix]
    func_detail = FuncDetail(fid, var_indices, "explicit", "binary functions", detail)
    functions.append(func_detail)
    
  # add factors to the GM
  for f in functions:
    n_indices = len(f.var_indices)
    if n_indices == 1:
      if verbose:
        print "  adding unary factor: {0}".format(f.detail)
        print "    fid: {0}   var: {1}".format(f.gm_function_id.getFunctionIndex(), f.var_indices[0])
      gm.addFactor(f.gm_function_id, f.var_indices[0])
    elif n_indices == 2:
      if verbose:
        print "  adding binary factor: {0}".format(f.detail)
        print "    fid: {0}   var: {1}".format(f.gm_function_id.getFunctionIndex(), f.var_indices)
      gm.addFactor(f.gm_function_id, f.var_indices)
    else:
      if verbose: print "skipping unexpected factor with {0} indices: {1}".format(n_indices, f.function_type)
  
  return gm, tracker



#===============================================================================
# Run belief proparation on a GM
#===============================================================================
def do_inference(gm, n_steps=120, damping=0., convergence_bound=0.001, verbose=False):
  """ Run belief propagation on the providede graphical model
  returns:
    energy (float): the energy of the GM
    var_indices (numpy array): indices for the best label for each variable
  """
  ogm_params = ogm.InfParam(steps=n_steps, damping=damping, convergenceBound=convergence_bound)
  infr_output = ogm.inference.BeliefPropagation(gm, parameter=ogm_params)
  
  if verbose:
    infr_output.infer(infr_output.verboseVisitor())
  else:
    infr_output.infer()
  
  detected_vars = []
  for i in range(0, gm.numberOfVariables):
    if gm.numberOfLabels(i) > 1:
      detected_vars.append(i)
  
  infr_marginals = infr_output.marginals(detected_vars)
  infr_marginals = np.exp(-infr_marginals)
  
  infr_best_match = infr_output.arg()
  infr_energy = infr_output.value()
  
  return infr_energy, infr_best_match, infr_marginals



def do_inference_astar(gm, heuristic='fast', accumulator='minimizer', verbose=False):
  infr_params = ogm.InfParam(heuristic=heuristic)
  infr = ogm.inference.AStar(gm=gm, accumulator=accumulator, parameter=infr_params)
  if verbose:
    infr.infer(infr.verboseVisitor())
  else:
    infr.infer()
  
  detected_vars = []
  for i in range(0, gm.numberOfVariables):
    if gm.numberOfLabels(i) > 1:
      detected_vars.append(i)
  
  #marginals = infr.marginals(detected_vars)
  best_match = infr.arg()
  energy = infr.value()
  
  return energy, best_match



#===============================================================================
# Generate a gm and run inference given an image and querygraph pair
#===============================================================================
"""def inference_pass(image_ix, ground_truth_ix, potentials, platt_models, vg_data, relationship_models, verbose=False):
  import time
  start = time.clock()
  
  query_graph = vg_data[ground_truth_ix].annotations

  object_detections = get_object_detections(image_ix, potentials, platt_models)
  attribute_detections = get_attribute_detections(image_ix, potentials, platt_models)

  image_filename = "/home/econser/School/Thesis/sg_dataset/sg_test_images/" + os.path.basename(vg_data[image_ix].image_path)

  gm, tracker = generate_pgm(query_graph, object_detections, attribute_detections, relationship_models, image_filename, verbose, autoplot=False)

  energy, indices, marginals = do_inference(gm)

  duration = time.clock() - start
  print "query {0} on image {1}: {2} sec".format(ground_truth_ix, image_ix, duration)

  return gm, tracker, energy, indices, marginals
"""


def gmm_pdf(X, mixture, mu, sigma):
  """ Gaussian Mixture Model PDF
  
  Given n (number of observations), m (number of features), c (number of components)
  
  Args:
    X : feature vector (n x m)
    mixture : GMM component vector (1 x c)
    mu : mu vectors (c x m)
    sigma : covariance matrices (c x m x m)
  
  returns:
    (n x 1) numpy vector of pdf values
  """
  n_components = len(mixture)
  n_vals = len(X)
  
  mixed_pdf = np.zeros(n_vals)
  for i in range(0, n_components):
    mixed_pdf += mvn.pdf(X, mu[i], sigma[i]) * mixture[i]
  
  return mixed_pdf



#===============================================================================
# Scenegraph generation functions
#
# query
#   objects: np.ndarray(n, dtype=object)
#     names
#   unary_triples:
#     subject: 1
#     predicate: 'is'
#     object: 'red'
#   binary_triples: np.ndarray(n, dytpe=object)
#     subject: 0
#     predicate: 'wearing'
#     object: 1
#===============================================================================
def gen_sro(query_str):
  import scipy.io.matlab.mio5_params as siom

  words = query_str.split()
  if len(words) != 3: return None
  
  sub_struct = siom.mat_struct()
  sub_struct.__setattr__('names', words[0])

  obj_struct = siom.mat_struct()
  obj_struct.__setattr__('names', words[2])

  rel_struct = siom.mat_struct()
  rel_struct.__setattr__('subject', 0)
  rel_struct.__setattr__('predicate', words[1])
  rel_struct.__setattr__('object', 1)
  
  det_list = np.array([sub_struct, obj_struct], dtype=np.object)
  query_struct = siom.mat_struct()
  query_struct.__setattr__('objects', det_list)
  query_struct.__setattr__('unary_triples', np.array([]))
  query_struct.__setattr__('binary_triples', rel_struct)

  return query_struct



def gen_asro(query_str):
  import scipy.io.matlab.mio5_params as siom

  words = query_str.split()
  if len(words) != 4: return None
  
  sub_attr_struct = siom.mat_struct()
  sub_attr_struct.__setattr__('subject', 0)
  sub_attr_struct.__setattr__('predicate', 'is')
  sub_attr_struct.__setattr__('object', words[0])
  
  query_struct = gen_sro(' '.join([words[1], words[2], words[3]]))
  query_struct.__setattr__('unary_triples', words[0])
  
  return query_struct



def gen_srao(query_str):
  import scipy.io.matlab.mio5_params as siom

  words = query_str.split()
  if len(words) != 4: return None
  
  obj_attr_struct = siom.mat_struct()
  obj_attr_struct.__setattr__('subject', 1)
  obj_attr_struct.__setattr__('predicate', 'is')
  obj_attr_struct.__setattr__('object', words[2])
  
  query_struct = gen_sro(' '.join([words[0], words[1], words[3]]))
  query_struct.__setattr__('unary_triples', words[2])
  
  return query_struct



def gen_asrao(query_str):
  import scipy.io.matlab.mio5_params as siom

  words = query_str.split()
  if len(words) != 5: return None
  
  obj_attr_struct = siom.mat_struct()
  obj_attr_struct.__setattr__('subject', 1)
  obj_attr_struct.__setattr__('predicate', 'is')
  obj_attr_struct.__setattr__('object', words[3])

  query_struct = gen_asro(' '.join([words[0], words[1], words[2], words[4]]))
  query_struct.__setattr__('unary_triples', obj_attr_struct)
  
  return query_struct

  
#===============================================================================
# Plotting functions
#===============================================================================
"""def draw_heatmap(image_filename, box_list, detection_values, marginal_values=[], title="", filename=""):
  import matplotlib.pyplot as plt
  import matplotlib.patches as patches
  from PIL import Image
  plt.switch_backend('Qt4Agg')
  
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
    plt.suptitle(title)
    
    plt.subplot(121)
    plt.imshow(img_array, cmap='gray')
    plt.imshow(d_map.T, alpha=0.3)
    plt.title("DETECTIONS")
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(122)
    plt.imshow(img_array, cmap='gray')
    plt.imshow(m_map.T, alpha=0.3)
    plt.title("MARGINALS")
    plt.xticks([])
    plt.yticks([])
  else:
    plt.figure(1)
    plt.xticks([])
    plt.yticks([])
    plt.title("DETECTIONS")
    plt.imshow(img_array)
    plt.imshow(d_map.T, alpha=0.3)
  
  plt.tight_layout()
  if len(filename) > 0:
    plt.savefig(filename, dpi=175)
  else:
    plt.show()



def draw_all_heatmaps(tracker, detections, marginals):
  for obj_ix in range(0, len(tracker.object_names)):
    obj_name = tracker.object_names[obj_ix]
    dets = detections['obj:'+obj_name]
    out_filename = obj_name+".png"
    draw_heatmap(tracker.image_path, dets[:,0:4], dets[:,4], marginals[obj_ix], filename=out_filename, title=obj_name)



def draw_each_box(filename, detections, do_heatmap=False):
  import matplotlib.pyplot as plt
  import matplotlib.patches as patches
  import matplotlib.lines as mlines
  import matplotlib.patheffects as path_effects
  from PIL import Image
  plt.switch_backend('Qt4Agg')
  
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




def draw_best_objects(tracker, indices, energy=None, filename="", image_size=[]):
  import randomcolor
  
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
    box_and_color = np.hstack((tracker.box_pairs[i][1][0:4], color_list[i]))
    box_list.append(box_and_color)
  
  title = "Object Detections"
  
  if energy != None:
    title = "Object Detections (energy={0:.3f})".format(energy)
  draw_image_box(tracker.image_path, box_list, legend_list, title, filename, size=image_size)



def draw_image_box(image_path, box_list, legend_list=[], relation_name="", filename="", size=[], verbose=False):
  import matplotlib.pyplot as plt
  import matplotlib.patches as patches
  import matplotlib.lines as mlines
  import matplotlib.patheffects as path_effects
  from PIL import Image
  plt.switch_backend('Qt4Agg')
  
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
    box = patches.Rectangle((x,y),w,h, linewidth=1, edgecolor=c, facecolor='none')
    ax.add_patch(box)
    txt = ax.text(x+5, y+5, legend_list[i][0], va='top', size=8, weight='bold', color='0.1')
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
    h = plt.Rectangle((0,0),0.25,0.25, fc=legend_list[i][1])
    handle_list.append(h)
    name_list.append(legend_list[i][0])
  plt.legend(handle_list, name_list)
  
  #plt.rcParams.update({'font.size': 6})
  plt.tight_layout(pad=1.25)
  
  if verbose: print '  dib: showing plot\r'
  if len(filename) == 0:
    plt.show()
  else:
    plt.savefig(filename, dpi=175)



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



"""
