from dataclasses import dataclass


@dataclass
class MappingManager:
    node_type_to_node_num = {
        "labitem": 753,
        "drug": 4294
    }
    node_type_to_node_feat_dim_in = {
        "admission": 8,
        "labitem": 2,
        "drug": 8
    }
    edge_type_to_edge_feat_dim_in = {
        ('admission', 'did', 'labitem'): 2,
        ('labitem', 'rev_did', 'admission'): 2,
        ("admission", "took", "drug"): 7,
        ("drug", "rev_took", "admission"): 7
    }


class HeteroGraphConfig:
    @staticmethod
    def use_all_edge_type():
        node_types = ['admission', 'labitem', 'drug']
        edge_types = [('admission', 'did', 'labitem'), ('labitem', 'rev_did', 'admission'),
                      ("admission", "took", "drug"), ("drug", "rev_took", "admission")]
        return node_types, edge_types

    @staticmethod
    def use_one_edge_type(item_type: str):
        r"""
        Args:
            - item_type: 'drug' or 'labitem'
        """
        if item_type == 'drug':
            node_types = ['admission', 'drug']
            edge_types = [("admission", "took", "drug"), ("drug", "rev_took", "admission")]
        elif item_type == 'labitem':
            node_types = ['admission', 'labitem']
            edge_types = [('admission', 'did', 'labitem'), ('labitem', 'rev_did', 'admission')]
        else:
            raise NotImplementedError

        return node_types, edge_types


if __name__ == "__main__":
    print(MappingManager.node_type_to_node_feat_dim_in['admission'])