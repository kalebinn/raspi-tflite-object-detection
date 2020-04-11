import os
class model():
    def __init__(self, model_directory):
        self.model_dir = model_directory
        self.graph = None
        self.labels = []
        
    def read_model(self, graph_file, labels_file):
        working_dir = os.getcwd() # current working directory
        
        # first we need to get the graph file
        self.graph = os.path.join(working_dir, self.model_dir, graph_file) # get the appropriate graph file
        print(self.graph)
        
        # next we need to get the labels
        label_file = os.path.join(working_dir, self.model_dir, labels_file)# get the cooresponding labelsmap
        with open(label_file, 'r') as f: 
            # read line by line, strip the new line character and make a list
            self.labels = [line.strip() for line in f.readlines()]
        
        #print(self.labels) # debug 
        
        # sometimes pretrained models will have "???" as a label for unknown objects, we need to delete them
        while (self.labels.count("???")):
            self.labels.remove("???")
        
        #print(self.labels) # check that ??? are gone
