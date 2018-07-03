import image_fetch_utils as ifu
import os.path

class ImageFetchDataset (object):
  def __init__(self, vg_data, potentials_data, platt_models, relationship_models, base_image_path):
    self.vg_data = vg_data
    self.potentials_data = potentials_data
    self.platt_models = platt_models
    self.relationship_models = relationship_models
    self.base_image_path = base_image_path
    
    self.current_image_num = -1
    self.object_detections = None
    self.attribute_detections = None
    self.per_object_attributes = None
    self.image_filename = ""
    self.current_sg_query = None

  def configure(self, test_image_num, sg_query):
    if test_image_num != self.current_image_num:
      self.current_image_num = test_image_num
      self.object_detections = ifu.get_object_detections(self.current_image_num, self.potentials_data, self.platt_models)
      self.attribute_detections = ifu.get_attribute_detections(self.current_image_num, self.potentials_data, self.platt_models)
      self.image_filename = self.base_image_path + os.path.basename(self.vg_data[self.current_image_num].image_path)
    
    if sg_query != self.current_sg_query:
      self.current_sg_query = sg_query
      self.per_object_attributes = ifu.get_object_attributes(self.current_sg_query)
