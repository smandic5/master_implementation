def get_selector_name(index: int) -> str:
    if index == 0:
        return "Uniform Selector"
    elif index == 1:
        return "Hard Task Selector"
    elif index == 2:
        return "INS Selector - Local Dissimilarity"
    elif index == 3:
        return "INS Selector - Global Dissimilarity"
    elif index == 4:
        return "INS Selector - Local Similarity"
    elif index == 5:
        return "INS Selector - Global Similarity"
    elif index == 6:
        return "Value Selector - Local Dissimilarity"
    elif index == 7:
        return "Value Selector - Global Dissimilarity"
    elif index == 8:
        return "Value Selector - Local Similarity"
    elif index == 9:
        return "Value Selector - Global Similarity"
    elif index == 10:
        return "Context Selector - Local Dissimilarity"
    elif index == 11:
        return "Context Selector - Global Dissimilarity"
    elif index == 12:
        return "Context Selector - Local Similarity"
    elif index == 13:
        return "Context Selector - Global Similarity"
    else:
        raise Exception(f"Unrecognized selector index: {index}")
    

def get_selector_old_name(index: int) -> str:
    if index == 0:
        return "UniformSelector"
    elif index == 1:
        return "HardTaskSelector"
    elif index == 2:
        return "InsSelector - Dissimilarity From Last"
    elif index == 3:
        return "InsSelector - Generic Dissimilarity"
    elif index == 4:
        return "InsSelector - Similarity From Last"
    elif index == 5:
        return "InsSelector - Generic Similarity"
    elif index == 6:
        return "ValueSelector - Dissimilarity From Last"
    elif index == 7:
        return "ValueSelector - Generic Dissimilarity"
    elif index == 8:
        return "ValueSelector - Similarity From Last"
    elif index == 9:
        return "ValueSelector - Generic Similarity"
    elif index == 10:
        return "ContextSelector - Dissimilarity From Last"
    elif index == 11:
        return "ContextSelector - Generic Dissimilarity"
    elif index == 12:
        return "ContextSelector - Similarity From Last"
    elif index == 13:
        return "ContextSelector - Generic Similarity"
    else:
        raise Exception(f"Unrecognized selector index: {index}")