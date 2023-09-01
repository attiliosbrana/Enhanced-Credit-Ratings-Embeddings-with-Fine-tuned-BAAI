prompt = """\
        The following text describes a financial methodology.
        
        ---------------------
        {context_str}
        ---------------------
        
        You are a financial analyst specialized in corporate credit ratings (credit analyst).
        Create {num_questions_per_chunk} complex questions based on this text \
        that would evaluate someone's deep understanding of this methodology. \
        Please keep your questions confined to the information in the text. \
        They can be either qualitative and quantitative in nature."
    """
