import csv
import os.path
import numpy as np
import opengm as ogm
import scipy.io as sio
import image_fetch_core as ifc; reload(ifc)
import image_fetch_plot as ifp; reload(ifp)



#===============================================================================
# Generate a gm and run inference given an image and querygraph pair
#===============================================================================
def image_batch(query, query_id, if_data, output_path, image_ix_subset=[], gen_plots=True, gm_method='original'):
  """  Run a single query against a batch of images

  Attributes:
    query (obj): the query to use on the images
    query_id (int) : id number of the query
    if_data (obj): ImageFetchData object
    output_path (str): "/home/econser/School/Thesis/data/inference test/"
    image_ix_subset (list): subset of image indices to run in the batch
  Returns:
    list of energy values from inference
  """
  
  time_list = []
  energy_list = []
  images = if_data.vg_data
  
  use_subset = False
  n_images = len(images)
  if len(image_ix_subset) > 0:
    use_subset = True
    n_images = len(image_ix_subset)
    
  for i in range(0, n_images):
    imgnum = i
    if use_subset: imgnum = image_ix_subset[i]
    
    gm, tracker, energy, best_matches, marginals, duration = inference_pass(query, query_id, imgnum, if_data, gm_method)
    
    energy_list.append(energy)
    time_list.append(duration)
    if gen_plots:
      filename = output_path + "q{:03d}".format(query_id) + "i{:03d}".format(imgnum) + ".png"
      ifp.draw_best_objects(tracker, best_matches, energy, filename)
  print "total time: {0}, average time: {1}".format(np.sum(time_list), np.average(time_list))
  
  file = open(output_path + "q{0}_energy_values.csv".format(query_id), "wb")
  csv_writer = csv.writer(file)
  csv_writer.writerow(("image_ix", "energy"))
  for i in range(0, n_images):
    imgnum = i
    if use_subset: imgnum = image_ix_subset[i]
    csv_writer.writerow((imgnum, energy_list[i]))
  file.close()



def query_batch(query_list, image_ix, if_data, output_path):
  """ Run a batch of queies against a single image

  Attributes:
    query_list (list): list of queries
    image_id (int): the image to run the queries against
    if_data (obj): and ImageFetchData object
    output_path (str): base path for output files "~/School/Thesis/data/inference test/"
  """
  
  energy_list = []
  time_list = []
  for query_ix in range(0, len(query_list)):
    #if_data.configure(image_id, query_list[query_ix])
    #gm, tracker = ifc.generate_pgm(if_data, False)
    #energy, matches, marginals = ifc.do_inference(gm)
    gm, tracker, energy, indices, marginals, duration = inference_pass(query_list[query_ix], query_ix, image_ix, if_data)
    
    energy_list.append(energy)
    time_list.append(duration)
    filename = output_path + "q" + str(query_ix) + "i" + str(image_id) + ".png"
    draw_best_objects(tracker, best_matches, energy, filename)
  print "total time: {0}, average time: {1}".format(np.sum(time_list), np.average(time_list))

  file = open(output_path+"energy_values.csv", "wb")
  csv_writer = csv.writer(file)
  csv_writer.writerow(("query_ix", "energy"))
  for i in range(0, len(query_list)):
    csv_writer.writerow((i, energy_list[i]))
  file.close()



def inference_pass(query, query_id, image_ix, if_data, gm_method='original'):
  """ Run the full inference pass using a query and image
  
  Attributes:
    query (obj): the query to execute
    query_id (int): a query identification number
    image_ix (int): the image to run against the image
    if_data (obj): an ImageFetchData object
  
  Returns:
    (object) opengm graphical model
    (object) DetectionTracker
    (float) energy value
    (numpy array) array of best-match indices
    (numpy array) post-inference marginals
    (float) total pass time
  """
  import time
  gm = None
  tracker = None
  start = time.time()  
  if_data.configure(image_ix, query)
  if gm_method == 'original':
    gm, tracker = ifc.generate_pgm(if_data, verbose=False)
  elif gm_method == 'uniform' or gm_method == 'empirical':
    gm, tracker = ifc.generate_pgm_all_objects(if_data, verbose=False, method=gm_method)
  energy, indices, marginals = ifc.do_inference(gm)
  duration = time.time() - start
  
  print "query {0} on image {1}: {2} sec".format(query_id, image_ix, duration)
  return gm, tracker, energy, indices, marginals, duration



