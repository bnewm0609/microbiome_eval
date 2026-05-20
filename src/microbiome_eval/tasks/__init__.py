def load_task(task_name):
    """
    Factory function to load a task by name.
    """
    if task_name == "disease_classification":
        from microbiome_eval.tasks.disease_clf import DiseaseClassificationTask
        return DiseaseClassification
    elif task_name == "microbiome_mcqa":
        from microbiome_eval.tasks.microbiome_mcqa import MicrobiomeMCQATask
        return MicrobiomeMCQATask
    elif task_name == "microbiome_litqa":
        from microbiome_eval.tasks.microbiome_litqa import MicrobiomeLitQATask
        return MicrobiomeLitQATask
    elif task_name == "med_qa":
        from microbiome_eval.tasks.medqa import MedQATask
        return MedQATask
    else:
        raise ValueError(f"Unknown task name: {task_name}")