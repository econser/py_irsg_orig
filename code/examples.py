#import matplotlib; matplotlib.use('Agg')
import matplotlib; matplotlib.use('Qt4Agg')
import data_pull as dp

import json
cfg_file = open('config.json')
cfg_data = json.load(cfg_file)
out_path = cfg_data['file_paths']['output_path']
img_path = cfg_data['file_paths']['image_path']
mat_path = cfg_data['file_paths']['mat_path']


"""
import image_fetch_core as ifc; reload(ifc)
import image_fetch_plot as ifp; reload(ifp)
import image_fetch_querygen as ifq; reload(ifq)
image_index = 234
query = ifq.gen_srao("man wearing red shirt")
ifdata.configure(image_index, query)
gm = None
tracker = None
verbose = False
gm, tracker = ifc.generate_pgm(ifdata, verbose)
energy = None
best_match_ix = None
marginals = None
energy, best_match_ix, marginals = ifc.do_inference(gm)
print("done")
"""

def ex1(query_index, image_index, inf_alg='bp', gm_method='original', do_suppl_plots=True, save_gm=False, verbose=True):
  """ generate plots for a query/image pair
  """
  import image_fetch_core as ifc; reload(ifc)
  import image_fetch_plot as ifp; reload(ifp)
  
  vgd, potentials, platt_mod, bin_mod, queries, ifdata = dp.get_all_data()
  
  query = vgd['vg_data_test'][query_index].annotations
  ifdata.configure(image_index, query)
  
  gm = None
  tracker = None
  if gm_method == 'original':
    gm, tracker = ifc.generate_pgm(ifdata, verbose)
  elif gm_method == 'uniform' or gm_method == 'empirical':
    gm, tracker = ifc.generate_pgm_all_objects(ifdata, verbose=verbose, method=gm_method)
  
  file_prefix = "q{0}_i{1}_".format(query_index, image_index)
  energy = None
  best_match_ix = None
  marginals = None
  obj_file = None
  
  if inf_alg == 'bp':
    obj_file = out_path + file_prefix + gm_method + '_bp_objects.png'
    energy, best_match_ix, marginals = ifc.do_inference(gm)
    if do_suppl_plots:
      ifp.draw_all_heatmaps(tracker, ifdata.object_detections, marginals, gm_method, out_path, file_prefix)
      #ifp.p_compare(tracker, ifdata.object_detections, marginals, out_path+file_prefix+'sctr.png')
  elif inf_alg == 'astar':
    obj_file = out_path + file_prefix + 'as_objects.png'
    energy, best_match_ix = ifc.do_inference_astar(gm)
  
  ifp.draw_best_objects(tracker, best_match_ix, energy, filename = obj_file)
  if save_gm: ifp.draw_gm(gm)



def ex2(query_index, image_index_list = [], gm_method='original'):
  """ batch - single query, multiple images
  """
  import os.path
  from datetime import datetime
  import image_fetch_wrappers as ifw
  
  vgd, potentials, platt_mod, bin_mod, queries, ifdata = dp.get_all_data()
  
  t = datetime.now()
  batch_path = 'q{0:03}_{1}{2:02}{3:02}_{4:02}{5:02}{6:02}/'.format(query_index, t.year, t.month, t.day, t.hour, t.minute, t.second)
  batch_path = out_path + batch_path
  if not os.path.exists(batch_path):
    os.makedirs(batch_path)
  
  query = vgd['vg_data_test'][query_index].annotations
  ifw.image_batch(query, query_index, ifdata, batch_path, image_index_list, gm_method)



def ex3(query_index_list, image_index_list = [], gm_method='original'):
  """ batch - multiple query, multiple images, only energy data saved
  """
  import os.path
  import image_fetch_wrappers as ifw; reload(ifw)
  
  vgd, potentials, platt_mod, bin_mod, queries, ifdata = dp.get_all_data()
  for query_index in query_index_list:
    query = vgd['vg_data_test'][query_index].annotations
    batch_path = out_path + gm_method + '/'
    if not os.path.exists(batch_path):
      os.makedirs(batch_path)
    ifw.image_batch(query, query_index, ifdata, batch_path, image_index_list, gm_method=gm_method, gen_plots=False)



def ex4(srao_query_string, image_index, gm_method='original', verbose=True):
  """ querygen
  """
  import image_fetch_core as ifc
  import image_fetch_utils as ifu; reload(ifu)
  import image_fetch_plot as ifp; reload(ifp)
  import image_fetch_querygen as ifq; reload(ifq)
  
  vgd, potentials, platt_mod, bin_mod, queries, ifdata = dp.get_all_data()
  
  query = ifq.gen_srao(srao_query_string)
  ifdata.configure(image_index, query)
  
  gm = None
  tracker = None
  if gm_method == 'original':
    gm, tracker = ifc.generate_pgm(ifdata, verbose)
  elif gm_method == 'uniform' or gm_method == 'empirical':
    gm, tracker = ifc.generate_pgm_all_objects(ifdata, verbose=verbose, method=gm_method)
  energy, best_match_ix, marginals = ifc.do_inference(gm)
  
  #ifp.draw_gm(gm)
  
  ifp.draw_best_objects(tracker, best_match_ix, energy, filename = out_path + "mq_i{0}.png".format(image_index))
  file_prefix = "mq_i{}_".format(image_index)
  ifp.draw_all_heatmaps(tracker, ifdata.object_detections, marginals, gm_method, out_path, file_prefix)
  



def ex5(query_index_list, image_index_list = [], gm_method='original', obscure_object=False):
  """ batch - multiple query, multiple images, only energy data saved
  """
  import os.path
  import image_fetch_wrappers as ifw; reload(ifw)
  import image_fetch_utils as ifu; reload(ifu)
  
  #if obscure_object and gm_method == 'original': return
  
  vgd, potentials, platt_mod, bin_mod, queries, ifdata = dp.get_all_data()
  batch_path = out_path + gm_method
  if obscure_object:
    batch_path = batch_path + '_obs_simple/'
  else:
    batch_path = batch_path + '_simple/'
  if not os.path.exists(batch_path):
    os.makedirs(batch_path)
  for query_index in query_index_list:
    query = queries['simple_graphs'][query_index].annotations
    if obscure_object:
      query = ifu.obscure(query)
    ifw.image_batch(query, query_index, ifdata, batch_path, image_index_list, gm_method=gm_method, gen_plots=False)



def alternate_inference_test(query_ix, image_ix, gm_method='original', verbose=True):
  import image_fetch_core as ifc
  import opengm as ogm
  vgd, potentials, platt_mod, bin_mod, queries, ifdata = dp.get_all_data()
  query = vgd['vg_data_test'][query_ix].annotations
  ifdata.configure(image_ix, query)
  gm = None
  tracker = None
  if gm_method == 'original':
    gm, tracker = ifc.generate_pgm(ifdata, verbose)
  elif gm_method == 'uniform' or gm_method == 'empirical':
    gm, tracker = ifc.generate_pgm_all_objects(ifdata, verbose, method=gm_method)
  #inf_param = ogm.InfParam(steps=120, damping=0., convergenceBound=0.001)
  #infr = ogm.inference.BeliefPropagation(gm, parameter=inf_param)
  #infr.infer(infr.verboseVisitor())
  return gm, tracker#, infr



#def gen_viz_file(query, query_id, query_str, image_set, tp_indices, ifdata, output_path):
def gen_viz_file(query_ixs, image_ixs, output_path):
  import image_fetch_core as ifc; reload(ifc)
  import image_fetch_utils as ifu; reload(ifu)
  
  vgd, potentials, platt_mod, bin_mod, queries, ifdata = dp.get_all_data()
  
  for query_ix in query_ixs:
    filename = 'q_' + str(query_ix) + '.csv'
    f = open(output_path+filename, 'w')
    
    query = queries['simple_graphs'][query_ix]
    query_str = ifu.sg_to_str(query.annotations)
    
    tp_simple = ifu.get_partial_scene_matches(vgd['vg_data_test'], queries['simple_graphs'])
    tp_indices = tp_simple[query_ix]
    
    for i in image_ixs: #range(0, len(image_ixs)):
      ifdata.configure(i, query.annotations)
      gm, tracker = ifc.generate_pgm(ifdata, verbose=False)
      energy, best_match_ix, marginals = ifc.do_inference(gm)
      
      line = '{:03d}, 0, "{}"\n'.format(i, query_str)
      f.write(line)
      line = '{:03d}, 2, {:0.4f}\n'.format(i, energy)
      f.write(line)
      
      if i in tp_indices:
        line = '{:03d}, 3, "match"\n'.format(i)
      else:
        line = '{:03d}, 3, "no match"\n'.format(i)
      f.write(line)
      
      for obj_ix in range(0, len(best_match_ix)):
        obj_name = tracker.object_names[obj_ix]
        box_ix = best_match_ix[obj_ix]
        bc = tracker.box_coords[box_ix]
        line = '{:03d}, 1, {}, "{}", {}, {}, {}, {}\n'.format(i, obj_ix, obj_name, int(bc[0]), int(bc[1]), int(bc[2]), int(bc[3]))
        f.write(line)
    
    f.close()
