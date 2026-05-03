from .utils import evaluate
from .mia import get_membership_attack_prob


def run_full_evaluation(model, retain_test_dl, forget_test_dl, retain_train_dl, forget_train_dl, test_dl, device, prefix=""):
    print(f"\n{prefix} Evaluation")
    print("Forget Test Acc :", evaluate(model, forget_test_dl, device))
    print("Retain Test Acc :", evaluate(model, retain_test_dl, device))
    mia = get_membership_attack_prob(retain_train_dl, forget_train_dl, test_dl, model)
    print("MIA Attack Acc  :", mia)
    return mia