import data_pull as dp

import json
cfg_file = open('config.json')
cfg_data = json.load(cfg_file)
out_path = cfg_data['file_paths']['output_path']
img_path = cfg_data['file_paths']['image_path']
mat_path = cfg_data['file_paths']['mat_path']

class InstanceScore (object):
    def __init__(self, name, probabilities, densities=None, nlls=None):
        self.name = name
        self.probs = probabilities
        self.densities = densities
        self.nlls = nlls

"""
import data_pull as dp
vgd, potentials, platt_mod, bin_mod, queries, ifdata = dp.get_all_data()

import image_fetch_querygen as ifq
query = ifq.gen_srao("man wearing red shirt")
reload(p); p.q_hist([234, 235, 236], 'man wearing shirt', ifdata)



q000 - clear glasses on woman
pos_0 = [164, 270, 336, 403, 449, 519, 635, 650]

q012 - sitting man holding phone
pos_12 = [65, 214, 394, 427, 449, 695, 744, 785, 920]

q137 - laptop on desk
pos_137 = [24, 45, 121, 204, 388, 407, 464, 662, 725, 835, 858, 907, 916]

import image_fetch_utils as ifu
ifu.findRelationship("man", "near", "dog", vgd)
ifu.getQueries(queries['simple_graphs'])
"""

def q_hist(images, sro_string, ifdata):
    import image_fetch_querygen as ifq
    query = ifq.gen_sro(sro_string)
    do_hist_(images, query, ifdata)



def do_hist_(image_nums, query, ifdata):
    import itertools
    import numpy as np
    import scipy.io as sio
    import matplotlib.pyplot as plt
    import image_fetch_core as ifc; reload(ifc)
    
    n_rows = len(image_nums)
    n_cols = len(query.objects) + 1
    plot_num = 1
    
    for image_num in image_nums:
        ifdata.configure(image_num, query)
        
        box_coords = []
        object_is_detected = []
        query_to_pgm = []
        pgm_to_query = []
        
        varcount = 0
        for obj_ix in range(0, len(ifdata.current_sg_query.objects)):
            query_object_name = ifdata.current_sg_query.objects[obj_ix].names
            
            # occasionally, there are multiple object names (is 0 the best?)
            if isinstance(query_object_name, np.ndarray):
                query_object_name = query_object_name[0]
            
            object_name = "obj:" + query_object_name
            if object_name not in ifdata.object_detections:
                object_is_detected.append(False)
                query_to_pgm.append(-1)
            else:
                if len(box_coords) == 0:
                    box_coords = np.copy(ifdata.object_detections[object_name][:,0:4])
                object_is_detected.append(True)
                query_to_pgm.append(varcount)
                varcount += 1
                pgm_to_query.append(obj_ix)
        
        #===========================================================================
        # UNARY PLOTS
        for i, o in enumerate(query.objects):
            object_name = o.names
            if isinstance(object_name, np.ndarray):
                object_name = object_name[0]
            scores = ifdata.object_detections['obj:' + object_name][:,4]
            title = '{} - n={}'.format(object_name, len(scores))
            plt.subplot(n_rows, n_cols, plot_num)
            hist_sub(scores, title=title, bins=50)
            plot_num += 1
        
        #===========================================================================
        # BINARY PLOTS
        box_cart_prod = np.array([x for x in itertools.product(box_coords, box_coords)])
        
        trip_root = ifdata.current_sg_query.binary_triples
        trip_list = []
        if isinstance(trip_root, sio.matlab.mio5_params.mat_struct):
            trip_list.append(ifdata.current_sg_query.binary_triples)
        else:
            for trip in trip_root:
                trip_list.append(trip)
            
        for trip in trip_list:
            sub_ix = trip.subject
            sub_pgm_ix = query_to_pgm[sub_ix]
            subject_name = ifdata.current_sg_query.objects[sub_ix].names
            if isinstance(subject_name, np.ndarray):
                subject_name = subject_name[0]
            
            obj_ix = trip.object
            obj_pgm_ix = query_to_pgm[obj_ix]
            object_name = ifdata.current_sg_query.objects[obj_ix].names
            if isinstance(object_name, np.ndarray):
                object_name = object_name[0]
            
            relationship = trip.predicate
            bin_trip_key = subject_name + "_" + relationship.replace(" ", "_")  + "_" + object_name
            
            # check if there is a gmm for the specific triple string
            if bin_trip_key not in ifdata.relationship_models:
                bin_trip_key = "*_" + relationship.replace(" ", "_") + "_*"
            if bin_trip_key not in ifdata.relationship_models:
                continue
            
            # verify object detections
            if sub_ix == obj_ix:
                continue
            
            if not object_is_detected[sub_ix]:
                continue
            
            if not object_is_detected[obj_ix]:
                continue
            
            # get model parameters
            prefix_object_name = "obj:" + object_name
            bin_object_box = ifdata.object_detections[prefix_object_name]
            
            prefix_subject_name = "obj:" + subject_name
            bin_subject_box = ifdata.object_detections[prefix_subject_name]
            
            rel_params = ifdata.relationship_models[bin_trip_key]
            
            # generate features from subject and object detection boxes
            sub_dim = 0
            obj_dim = 1
            
            subx_center = box_cart_prod[:, sub_dim, 0] + 0.5 * box_cart_prod[:, sub_dim, 2]
            suby_center = box_cart_prod[:, sub_dim, 1] + 0.5 * box_cart_prod[:, sub_dim, 3]
            
            objx_center = box_cart_prod[:, obj_dim, 0] + 0.5 * box_cart_prod[:, obj_dim, 2]
            objy_center = box_cart_prod[:, obj_dim, 1] + 0.5 * box_cart_prod[:, obj_dim, 3]
            
            sub_width = box_cart_prod[:, sub_dim, 2]
            relx_center = (subx_center - objx_center) / sub_width
            
            sub_height = box_cart_prod[:, sub_dim, 3]
            rely_center = (suby_center - objy_center) / sub_height
            
            rel_height = box_cart_prod[:, obj_dim, 2] / box_cart_prod[:, sub_dim, 2]
            rel_width = box_cart_prod[:, obj_dim, 3] / box_cart_prod[:, sub_dim, 3]
            
            ftrs = np.vstack((relx_center, rely_center, rel_height, rel_width)).T
            
            scores = gmm_pdf(ftrs, rel_params.gmm_weights, rel_params.gmm_mu, rel_params.gmm_sigma)
            eps = np.finfo(np.float).eps
            scores = np.log(eps + scores)
            sig_scores = 1.0 / (1. + np.exp(rel_params.platt_a * scores + rel_params.platt_b))
            
            title='{} - {}'.format(bin_trip_key, len(sig_scores))
            
            plt.subplot(n_rows, n_cols, plot_num)
            hist_sub(sig_scores, title, bins=50)
            plot_num += 1
    
    plt.show()



def hist_sub(scores, title='', bins=100):
    import numpy as np
    import matplotlib.pyplot as plt
    
    bins_ = np.linspace(start=0., stop=1., num=bins)
    n, bins, patches = plt.hist(scores, bins=bins_, log=True)
    
    axes = plt.gca()
    y_max = np.max(n)
    y_max = np.log10(y_max).round()+1
    if y_max < 3:
        y_max = 3
        
    
    axes.set_ylim([0.1, pow(10,y_max)])
    axes.set_xlim([0.0, 1.0])
    
    if len(title) != 0:
        plt.title(title)
    
    plt.grid(True)



def do_hist(image_num, query, ifdata):
    import image_fetch_core as ifc; reload(ifc)
    import scipy.io as sio
    import numpy as np
    import itertools
    
    ifdata.configure(image_num, query)
    
    box_coords = []
    object_is_detected = []
    query_to_pgm = []
    pgm_to_query = []
    
    varcount = 0
    for obj_ix in range(0, len(ifdata.current_sg_query.objects)):
        query_object_name = ifdata.current_sg_query.objects[obj_ix].names
        
        # occasionally, there are multiple object names (is 0 the best?)
        if isinstance(query_object_name, np.ndarray):
            query_object_name = query_object_name[0]
        
        object_name = "obj:" + query_object_name
        if object_name not in ifdata.object_detections:
            object_is_detected.append(False)
            query_to_pgm.append(-1)
        else:
            if len(box_coords) == 0:
                box_coords = np.copy(ifdata.object_detections[object_name][:,0:4])
            object_is_detected.append(True)
            query_to_pgm.append(varcount)
            varcount += 1
            pgm_to_query.append(obj_ix)
    
    #===========================================================================
    # UNARY PLOTS
    for i, o in enumerate(query.objects):
        object_name = o.names
        if isinstance(object_name, np.ndarray):
            object_name = object_name[0]
        scores = ifdata.object_detections['obj:' + object_name][:,4]
        title = '{} - n={}'.format(object_name, len(scores))
        hist(scores, title=title, bins=50)
    
    #===========================================================================
    # BINARY PLOTS
    box_cart_prod = np.array([x for x in itertools.product(box_coords, box_coords)])
    
    trip_root = ifdata.current_sg_query.binary_triples
    trip_list = []
    if isinstance(trip_root, sio.matlab.mio5_params.mat_struct):
        trip_list.append(ifdata.current_sg_query.binary_triples)
    else:
        for trip in trip_root:
            trip_list.append(trip)
    
    for trip in trip_list:
        sub_ix = trip.subject
        sub_pgm_ix = query_to_pgm[sub_ix]
        subject_name = ifdata.current_sg_query.objects[sub_ix].names
        if isinstance(subject_name, np.ndarray):
            subject_name = subject_name[0]
        
        obj_ix = trip.object
        obj_pgm_ix = query_to_pgm[obj_ix]
        object_name = ifdata.current_sg_query.objects[obj_ix].names
        if isinstance(object_name, np.ndarray):
            object_name = object_name[0]
        
        relationship = trip.predicate
        bin_trip_key = subject_name + "_" + relationship.replace(" ", "_")  + "_" + object_name
        
        # check if there is a gmm for the specific triple string
        if bin_trip_key not in ifdata.relationship_models:
            bin_trip_key = "*_" + relationship.replace(" ", "_") + "_*"
        if bin_trip_key not in ifdata.relationship_models:
            continue
        
        # verify object detections
        if sub_ix == obj_ix:
            continue
        
        if not object_is_detected[sub_ix]:
            continue
        
        if not object_is_detected[obj_ix]:
            continue
        
        # get model parameters
        prefix_object_name = "obj:" + object_name
        bin_object_box = ifdata.object_detections[prefix_object_name]
        
        prefix_subject_name = "obj:" + subject_name
        bin_subject_box = ifdata.object_detections[prefix_subject_name]
        
        rel_params = ifdata.relationship_models[bin_trip_key]
        
        # generate features from subject and object detection boxes
        sub_dim = 0
        obj_dim = 1
        
        subx_center = box_cart_prod[:, sub_dim, 0] + 0.5 * box_cart_prod[:, sub_dim, 2]
        suby_center = box_cart_prod[:, sub_dim, 1] + 0.5 * box_cart_prod[:, sub_dim, 3]
        
        objx_center = box_cart_prod[:, obj_dim, 0] + 0.5 * box_cart_prod[:, obj_dim, 2]
        objy_center = box_cart_prod[:, obj_dim, 1] + 0.5 * box_cart_prod[:, obj_dim, 3]
        
        sub_width = box_cart_prod[:, sub_dim, 2]
        relx_center = (subx_center - objx_center) / sub_width
        
        sub_height = box_cart_prod[:, sub_dim, 3]
        rely_center = (suby_center - objy_center) / sub_height
        
        rel_height = box_cart_prod[:, obj_dim, 2] / box_cart_prod[:, sub_dim, 2]
        rel_width = box_cart_prod[:, obj_dim, 3] / box_cart_prod[:, sub_dim, 3]
        
        ftrs = np.vstack((relx_center, rely_center, rel_height, rel_width)).T
        
        scores = gmm_pdf(ftrs, rel_params.gmm_weights, rel_params.gmm_mu, rel_params.gmm_sigma)
        eps = np.finfo(np.float).eps
        scores = np.log(eps + scores)
        sig_scores = 1.0 / (1. + np.exp(rel_params.platt_a * scores + rel_params.platt_b))
        
        title='{} - {}'.format(bin_trip_key, len(sig_scores))
        hist(sig_scores, title, bins=50)



def hist(scores, title='', bins=100):
    import numpy as np
    import matplotlib.pyplot as plt
    
    bins_ = np.linspace(start=0., stop=1., num=bins)
    n, bins, patches = plt.hist(scores, bins=bins_, log=True)
    
    axes = plt.gca()
    y_max = np.max(n)
    y_max = np.log10(y_max).round()+1
    if y_max < 3:
        y_max = 3
        
    
    axes.set_ylim([0.1, pow(10,y_max)])
    axes.set_xlim([0.0, 1.0])
    
    if len(title) != 0:
        plt.title(title)
    
    plt.grid(True)
    plt.show()



def gmm_pdf(X, mixture, mu, sigma):
    import numpy as np
    from scipy.stats import multivariate_normal as mvn
    
    n_components = len(mixture)
    n_vals = len(X)
    
    mixed_pdf = np.zeros(n_vals)
    for i in range(0, n_components):
        mixed_pdf += mvn.pdf(X, mu[i], sigma[i]) * mixture[i]
    
    return mixed_pdf



################################################################################
def avg_plot(image_nums, sro_string, ifdata):
    import matplotlib.pyplot as plt
    
    sum_of_bins, bin_names, bin_ranges = avg_hist(image_nums, sro_string, ifdata)
    
    n_rows = 1
    n_cols = len(bin_names)
    plot_num = 1
    
    for i, hist_data in enumerate(sum_of_bins):
        plt.subplot(n_rows, n_cols, plot_num)
        plt.bar(bin_ranges[:-1], hist_data, width=0.02, log=True)
        ax = plt.gca()
        ax.set_ylim([0.1, pow(10,3)])
        ax.set_xlim([0.0, 1.0])
        plt.title(bin_names[i])
        plt.grid(True)
        plot_num += 1
    
    plt.subplots_adjust(wspace=0.20)
    plt.show()



def avg_hist(image_nums, sro_string, ifdata):
    import itertools
    import numpy as np
    import scipy.io as sio
    import matplotlib.pyplot as plt
    import image_fetch_core as ifc; reload(ifc)
    
    import image_fetch_querygen as ifq
    query = ifq.gen_sro(sro_string)
    
    n_objects = 1
    if isinstance(query.objects, np.ndarray):
        n_objects = len(query.objects)
    
    n_relations = 1
    if isinstance(query.binary_triples, np.ndarray):
        n_relations = len(query.binary_triples)
    
    n_cols = n_objects + n_relations
    n_rows = 1
    plot_num = 1
    
    for image_num in image_nums:
        ifdata.configure(image_num, query)
        
        box_coords = []
        object_is_detected = []
        query_to_pgm = []
        pgm_to_query = []
        
        sum_of_bins = np.zeros((n_cols,50))
        bin_names = []
        bin_ranges = None
        
        varcount = 0
        for obj_ix in range(0, len(ifdata.current_sg_query.objects)):
            query_object_name = ifdata.current_sg_query.objects[obj_ix].names
            
            # occasionally, there are multiple object names (is 0 the best?)
            if isinstance(query_object_name, np.ndarray):
                query_object_name = query_object_name[0]
            
            object_name = "obj:" + query_object_name
            if object_name not in ifdata.object_detections:
                object_is_detected.append(False)
                query_to_pgm.append(-1)
            else:
                if len(box_coords) == 0:
                    box_coords = np.copy(ifdata.object_detections[object_name][:,0:4])
                object_is_detected.append(True)
                query_to_pgm.append(varcount)
                varcount += 1
                pgm_to_query.append(obj_ix)
        
        #===========================================================================
        # UNARY PROBABILITIES
        for i, o in enumerate(query.objects):
            object_name = o.names
            if isinstance(object_name, np.ndarray):
                object_name = object_name[0]
            scores = ifdata.object_detections['obj:' + object_name][:,4]
            
            histo = np.histogram(scores, bins=50, range=(0.0, 1.0))
            sum_of_bins[i] += histo[0]
            
            if bin_ranges is None:
                bin_ranges = histo[1]
            
            if len(bin_names) != n_cols:
                bin_names.append(o.names)
        
        #===========================================================================
        # BINARY PLOTS
        box_cart_prod = np.array([x for x in itertools.product(box_coords, box_coords)])
        
        trip_root = ifdata.current_sg_query.binary_triples
        trip_list = []
        if isinstance(trip_root, sio.matlab.mio5_params.mat_struct):
            trip_list.append(ifdata.current_sg_query.binary_triples)
        else:
            for trip in trip_root:
                trip_list.append(trip)
            
        for i, trip in enumerate(trip_list):
            sub_ix = trip.subject
            sub_pgm_ix = query_to_pgm[sub_ix]
            subject_name = ifdata.current_sg_query.objects[sub_ix].names
            if isinstance(subject_name, np.ndarray):
                subject_name = subject_name[0]
            
            obj_ix = trip.object
            obj_pgm_ix = query_to_pgm[obj_ix]
            object_name = ifdata.current_sg_query.objects[obj_ix].names
            if isinstance(object_name, np.ndarray):
                object_name = object_name[0]
            
            relationship = trip.predicate
            bin_trip_key = subject_name + "_" + relationship.replace(" ", "_")  + "_" + object_name
            
            # check if there is a gmm for the specific triple string
            if bin_trip_key not in ifdata.relationship_models:
                bin_trip_key = "*_" + relationship.replace(" ", "_") + "_*"
            if bin_trip_key not in ifdata.relationship_models:
                continue
            
            # verify object detections
            if sub_ix == obj_ix:
                continue
            
            if not object_is_detected[sub_ix]:
                continue
            
            if not object_is_detected[obj_ix]:
                continue
            
            # get model parameters
            prefix_object_name = "obj:" + object_name
            bin_object_box = ifdata.object_detections[prefix_object_name]
            
            prefix_subject_name = "obj:" + subject_name
            bin_subject_box = ifdata.object_detections[prefix_subject_name]
            
            rel_params = ifdata.relationship_models[bin_trip_key]
            
            # generate features from subject and object detection boxes
            sub_dim = 0
            obj_dim = 1
            
            subx_center = box_cart_prod[:, sub_dim, 0] + 0.5 * box_cart_prod[:, sub_dim, 2]
            suby_center = box_cart_prod[:, sub_dim, 1] + 0.5 * box_cart_prod[:, sub_dim, 3]
            
            objx_center = box_cart_prod[:, obj_dim, 0] + 0.5 * box_cart_prod[:, obj_dim, 2]
            objy_center = box_cart_prod[:, obj_dim, 1] + 0.5 * box_cart_prod[:, obj_dim, 3]
            
            sub_width = box_cart_prod[:, sub_dim, 2]
            relx_center = (subx_center - objx_center) / sub_width
            
            sub_height = box_cart_prod[:, sub_dim, 3]
            rely_center = (suby_center - objy_center) / sub_height
            
            rel_height = box_cart_prod[:, obj_dim, 2] / box_cart_prod[:, sub_dim, 2]
            rel_width = box_cart_prod[:, obj_dim, 3] / box_cart_prod[:, sub_dim, 3]
            
            ftrs = np.vstack((relx_center, rely_center, rel_height, rel_width)).T
            
            scores = gmm_pdf(ftrs, rel_params.gmm_weights, rel_params.gmm_mu, rel_params.gmm_sigma)
            eps = np.finfo(np.float).eps
            scores = np.log(eps + scores)
            sig_scores = 1.0 / (1. + np.exp(rel_params.platt_a * scores + rel_params.platt_b))
            
            if len(bin_names) != n_cols:
                bin_names.append(bin_trip_key)
            
            histo = np.histogram(sig_scores, bins=50, range=(0.0, 1.0))
            sum_of_bins[i + n_objects] += histo[0]
    
    return sum_of_bins, bin_names, bin_ranges
