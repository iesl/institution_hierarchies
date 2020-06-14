
import os
import pickle

import main.utils.data_pathnames as data_pathnames


'''
DataExtractor objects to represent getting different formats of data 
'''
class DataExtractor(object):

    def __init__(self, config):
        '''
        Sets the right flags for data to collect

        :param config:
        '''
        self.config = config
        data_pathnames.init()

        self.inst_name_dict = pickle.load(open(data_pathnames.GRID2INST_NAME, 'rb'))
        self.city_dict = pickle.load(open(data_pathnames.GRID2CITY, 'rb'))
        self.state_dict = pickle.load(open(data_pathnames.GRID2STATE, 'rb'))
        self.country_dict = pickle.load(open(data_pathnames.GRID2COUNTRY, 'rb'))
        self.type_dict = pickle.load(open(data_pathnames.GRID2TYPE, 'rb'))


    def get_batch_data(self, batch_ids, is_train=True):
        '''
        Extract correct data from batch

        :param batch_ids: batch of ids (qry_id, pos_id, neg_id) or (qry_id, cnd_id, lbl)
        :param is_train:
        :return:
        '''
        batch_data = []

        if is_train:
            for (qry_id, pos_id, neg_id) in batch_ids:
                qry_data = []
                pos_data = []
                neg_data = []

                # Add institution data
                qry_data.append(self.inst_name_dict[qry_id])
                pos_data.append(self.inst_name_dict[pos_id])
                neg_data.append(self.inst_name_dict[neg_id])

                # Add city to data
                if self.config.include_city:
                    qry_data.append(self.city_dict[qry_id])
                    pos_data.append(self.city_dict[pos_id])
                    neg_data.append(self.city_dict[neg_id])

                # Add state to data
                if self.config.include_state:
                    qry_data.append(self.state_dict[qry_id])
                    pos_data.append(self.state_dict[pos_id])
                    neg_data.append(self.state_dict[neg_id])

                # Add country to data
                if self.config.include_country:
                    qry_data.append(self.country_dict[qry_id])
                    pos_data.append(self.country_dict[pos_id])
                    neg_data.append(self.country_dict[neg_id])

                if self.config.include_type:
                    if qry_id in self.type_dict:
                        qry_data.append(' '.join(self.type_dict[qry_id]))
                    else:
                        qry_data.append("")
                    if pos_id in self.type_dict:
                        pos_data.append(' '.join(self.type_dict[pos_id]))
                    else:
                        pos_data.append("")
                    if neg_id in self.type_dict:
                        neg_data.append(' '.join(self.type_dict[neg_id]))
                    else:
                        neg_data.append("")

                batch_data.append([qry_data, pos_data, neg_data])

        else:
            for (qry_id, cnd_id) in list(zip(*batch_ids)):
                qry_data = []
                cnd_data = []

                # Add institution data
                qry_data.append(self.inst_name_dict[qry_id])
                cnd_data.append(self.inst_name_dict[cnd_id])

                # Add city to data
                if self.config.include_city:
                    qry_data.append(self.city_dict[qry_id])
                    cnd_data.append(self.city_dict[cnd_id])

                # Add state to data
                if self.config.include_state:
                    qry_data.append(self.state_dict[qry_id])
                    cnd_data.append(self.state_dict[cnd_id])

                # Add country to data
                if self.config.include_country:
                    qry_data.append(self.country_dict[qry_id])
                    cnd_data.append(self.country_dict[cnd_id])

                if self.config.include_type:
                    if qry_id in self.type_dict:
                        qry_data.append(' '.join(self.type_dict[qry_id]))
                    else:
                        qry_data.append("")
                    if cnd_id in self.type_dict:
                        cnd_data.append(' '.join(self.type_dict[cnd_id]))
                    else:
                        cnd_data.append("")

                batch_data.append([qry_data, cnd_data])

        return batch_data


