# -*- coding: utf-8 -*-


from src.data_processing.group_manipulation import create_combinations



def create_configurations(configurations_base, optional_products):
	optional_configurations = create_combinations(optional_products, root = False)

	all_configurations = []

	for base in configurations_base:
		all_configurations.append(base)
		for optional in optional_configurations:
		    all_configurations.append(base + optional)








if __name__ == "__main__":
	create_configurations()
	
	
