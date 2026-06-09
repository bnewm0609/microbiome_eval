def load_task(task_name):
    """
    Factory function to load a task by name.
    """
    if task_name == "disease_classification":
        from microbiome_eval.tasks.disease_clf import DiseaseClassificationTask
        return DiseaseClassificationTask
    elif task_name == "microbiome_reasoning":
        from microbiome_eval.tasks.microbiome_reasoning import MicrobiomeReasoningTask
        return MicrobiomeReasoningTask
    elif task_name == "microbiome_litqa":
        from microbiome_eval.tasks.microbiome_litqa import MicrobiomeLitQA
        return MicrobiomeLitQA
    elif task_name == "med_qa":
        from microbiome_eval.tasks.medqa import MedQATask
        return MedQATask
    elif task_name == "hard_microbiome_qs":
        from microbiome_eval.tasks.hard_microbiome_qs import HardMicrobiomeQsTask
        return HardMicrobiomeQsTask
    elif task_name == "methods_errors":
        from microbiome_eval.tasks.methods_errors import MethodsErrors
        return MethodsErrors
    else:
        raise ValueError(f"Unknown task name: {task_name}")