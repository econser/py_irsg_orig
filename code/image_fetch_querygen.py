import numpy as np
import scipy.io.matlab.mio5_params as siom

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
  words = query_str.split()
  if len(words) != 4: return None
  
  sub_attr_struct = siom.mat_struct()
  sub_attr_struct.__setattr__('subject', 0)
  sub_attr_struct.__setattr__('predicate', 'is')
  sub_attr_struct.__setattr__('object', words[0])
  
  query_struct = gen_sro(' '.join([words[1], words[2], words[3]]))
  query_struct.__setattr__('unary_triples', sub_attr_struct)
  
  return query_struct



def gen_srao(query_str):
  words = query_str.split()
  if len(words) != 4: return None
  
  obj_attr_struct = siom.mat_struct()
  obj_attr_struct.__setattr__('subject', 1)
  obj_attr_struct.__setattr__('predicate', 'is')
  obj_attr_struct.__setattr__('object', words[2])
  
  query_struct = gen_sro(' '.join([words[0], words[1], words[3]]))
  query_struct.__setattr__('unary_triples', obj_attr_struct)
  
  return query_struct



def gen_asrao(query_str):
  words = query_str.split()
  if len(words) != 5: return None
  
  obj_attr_struct = siom.mat_struct()
  obj_attr_struct.__setattr__('subject', 1)
  obj_attr_struct.__setattr__('predicate', 'is')
  obj_attr_struct.__setattr__('object', words[3])

  query_struct = gen_asro(' '.join([words[0], words[1], words[2], words[4]]))
  query_struct.__setattr__('unary_triples', obj_attr_struct)
  
  return query_struct



"""
cl = sio.loadmat('/home/econser/School/Thesis/code/model_params/not_used/class_lists.mat', struct_as_record=False, squeeze_me=True)
obj_names = cl['class_lists'].object_names
atr_names = cl['class_lists'].attribute_names
rel_names = cl['class_lists'].relationship_names
gen_scene(5, 2, 4, obj_names, atr_names, rel_names)
"""
def gen_scene(n_objects, unary_per_object, bin_per_object, object_names, unary_names, binary_names):
  """ Generate a random scenegraph
  As anticipated, the scenes are completely nonsensical
  """
  obj_shuffle = np.arange(len(object_names))
  np.random.shuffle(obj_shuffle)
  obj_selections = obj_shuffle[0:n_objects]
  objects = object_names[obj_selections]
  
  for obj_ix in range(0, n_objects):
    # gen unaries
    shuffleix = np.arange(len(unary_names))
    np.random.shuffle(shuffleix)
    indices = shuffleix[0:unary_per_object]
    unaries = unary_names[indices]
    # gen binaries
    shuffleix = np.arange(len(binary_names))
    np.random.shuffle(shuffleix)
    indices = shuffleix[0:bin_per_object]
    binaries = binary_names[indices]
    binary_obj_ix = np.random.randint(0, n_objects-1, size=bin_per_object)
    for i in range(0, bin_per_object):
      if binary_obj_ix[i] >= obj_ix:
        binary_obj_ix[i] += 1
    binary_objects = objects[binary_obj_ix]
    
    print '{}: {}'.format(objects[obj_ix], unaries)
    for i in range(0, bin_per_object):
      print '    {} {}'.format(binaries[i], binary_objects[i])
