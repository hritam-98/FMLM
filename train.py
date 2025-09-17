import torch
import torch.nn as nn
import torch.optim as optim
from models import Encoder, Classifier, ReconstructionNetwork
from data_loader import get_data_loader
from utils import EntropyLoss, kl_divergence_loss, reparameterize
import argparse
from tqdm import tqdm
import itertools

def train_stage1(args, device):
    """
    Stage 1: Learn Domain-specific Features
    Train (E_S, C_S) and (E_T, C_T) pairs.
    """
    print("--- Starting Stage 1: Learning Domain-specific Features ---")

    # --- Initialize Models ---
    # Source-specific models
    encoder_S = Encoder(num_classes=args.num_classes).to(device)
    classifier_S = Classifier(num_classes=args.num_classes).to(device)
    # Target-specific models
    encoder_T = Encoder(num_classes=args.num_classes).to(device)
    classifier_T = Classifier(num_classes=args.num_classes).to(device)

    # --- Optimizers ---
    optimizer_S = optim.AdamW(
        list(encoder_S.parameters()) + list(classifier_S.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    optimizer_T = optim.AdamW(
        list(encoder_T.parameters()) + list(classifier_T.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )

    # --- Loss Functions ---
    classification_loss_fn = nn.CrossEntropyLoss()
    uncertainty_loss_fn = EntropyLoss()

    # --- DataLoaders ---
    source_loader = get_data_loader(args.data_path, args.source, batch_size=args.batch_size, is_train=True)
    target_labeled_loader, target_unlabeled_loader = get_data_loader(
        args.data_path, args.target, batch_size=args.batch_size, is_train=True, num_labeled_per_class=args.num_labeled
    )
    # For stage 1, we combine labeled and unlabeled target data for the uncertainty loss part
    target_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset([target_labeled_loader.dataset, target_unlabeled_loader.dataset]),
        batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True
    )


    # --- Training Loop ---
    for epoch in range(args.epochs_stage1):
        # Using tqdm for progress bars
        loop = tqdm(zip(source_loader, target_loader, target_labeled_loader), total=min(len(source_loader), len(target_loader)))
        total_loss_s = 0
        total_loss_t = 0
        
        for (source_imgs, source_labels), (target_imgs, _), (target_labeled_imgs, target_labeled_labels) in loop:
            source_imgs, source_labels = source_imgs.to(device), source_labels.to(device)
            target_imgs = target_imgs.to(device)
            target_labeled_imgs, target_labeled_labels = target_labeled_imgs.to(device), target_labeled_labels.to(device)
            
            # --- Train Source Models (E_S, C_S) ---
            optimizer_S.zero_grad()
            mu_s, log_var_s = encoder_S(source_imgs)
            z_s = reparameterize(mu_s, log_var_s)
            preds_s = classifier_S(z_s)
            
            # In-domain classification loss (Eq. 1)
            l_c_s = classification_loss_fn(preds_s, source_labels)
            
            # Out-of-domain uncertainty loss (Eq. 2 & 3)
            mu_t_for_s, log_var_t_for_s = encoder_S(target_imgs)
            z_t_for_s = reparameterize(mu_t_for_s, log_var_t_for_s)
            preds_t_on_s = classifier_S(z_t_for_s)
            l_u_s = uncertainty_loss_fn(preds_t_on_s)

            # Total source loss (Eq. 4)
            loss_s = l_c_s + args.alpha * l_u_s 
            loss_s.backward()
            optimizer_S.step()
            total_loss_s += loss_s.item()

            # --- Train Target Models (E_T, C_T) ---
            optimizer_T.zero_grad()
            mu_t, log_var_t = encoder_T(target_labeled_imgs)
            z_t = reparameterize(mu_t, log_var_t)
            preds_t = classifier_T(z_t)

            # In-domain classification loss (Eq. 1) - only on labeled target data
            l_c_t = classification_loss_fn(preds_t, target_labeled_labels)

            # Out-of-domain uncertainty loss (Eq. 2 & 3)
            mu_s_for_t, log_var_s_for_t = encoder_T(source_imgs)
            z_s_for_t = reparameterize(mu_s_for_t, log_var_s_for_t)
            preds_s_on_t = classifier_T(z_s_for_t)
            l_u_t = uncertainty_loss_fn(preds_s_on_t)

            # Total target loss (Eq. 4)
            loss_t = l_c_t + args.alpha * l_u_t
            loss_t.backward()
            optimizer_T.step()
            total_loss_t += loss_t.item()

            loop.set_description(f"Epoch [{epoch+1}/{args.epochs_stage1}]")
            loop.set_postfix(loss_S=loss_s.item(), loss_T=loss_t.item())

    print("--- Stage 1 Finished ---")
    return encoder_S, classifier_S, encoder_T, classifier_T


def train_stage2(args, device, encoder_S, classifier_S, encoder_T, classifier_T):
    """
    Stage 2: Forget and Learn More
    Train Reconstruction network (R) and Domain-agnostic classifier (C).
    """
    print("\n--- Starting Stage 2: Forgetting and Learning More ---")

    # --- Freeze Stage 1 models ---
    for param in encoder_S.parameters(): param.requires_grad = False
    for param in classifier_S.parameters(): param.requires_grad = False
    for param in encoder_T.parameters(): param.requires_grad = False
    for param in classifier_T.parameters(): param.requires_grad = False
    encoder_S.eval()
    classifier_S.eval()
    encoder_T.eval()
    classifier_T.eval()

    # --- Initialize Models for Stage 2 ---
    recon_net = ReconstructionNetwork().to(device)
    domain_agnostic_classifier = Classifier(num_classes=args.num_classes).to(device)

    # --- Optimizer ---
    optimizer = optim.AdamW(
        list(recon_net.parameters()) + list(domain_agnostic_classifier.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )

    # --- Loss Functions ---
    recon_loss_fn = nn.MSELoss()
    unlearning_loss_fn = EntropyLoss()
    classification_loss_fn = nn.CrossEntropyLoss()

    # --- DataLoaders ---
    source_loader = get_data_loader(args.data_path, args.source, batch_size=args.batch_size, is_train=True)
    target_labeled_loader, target_unlabeled_loader = get_data_loader(
        args.data_path, args.target, batch_size=args.batch_size, is_train=True, num_labeled_per_class=args.num_labeled
    )

    # --- Training Loop ---
    for epoch in range(args.epochs_stage2):
        # Use itertools.cycle to loop over the shorter dataloaders
        loaders = [source_loader, target_labeled_loader, target_unlabeled_loader]
        num_batches = max(len(loader) for loader in loaders)
        data_iter = zip(*(itertools.cycle(loader) for loader in loaders))
        
        loop = tqdm(data_iter, total=num_batches)
        
        for (source_imgs, source_labels), (target_labeled_imgs, target_labels), (target_unlabeled_imgs, _) in loop:
            source_imgs, source_labels = source_imgs.to(device), source_labels.to(device)
            target_labeled_imgs, target_labels = target_labeled_imgs.to(device), target_labels.to(device)
            target_unlabeled_imgs = target_unlabeled_imgs.to(device)
            
            all_target_imgs = torch.cat([target_labeled_imgs, target_unlabeled_imgs], dim=0)

            optimizer.zero_grad()

            # --- Forget (Unlearn) Step ---
            # Reconstruct images (Eq. 5)
            recon_s = recon_net(source_imgs)
            recon_t = recon_net(all_target_imgs)
            
            l_r_s = recon_loss_fn(recon_s, source_imgs)
            l_r_t = recon_loss_fn(recon_t, all_target_imgs)
            l_r = l_r_s + l_r_t

            # Unlearning loss (Eq. 6)
            with torch.no_grad():
                mu_s, log_var_s = encoder_S(recon_s)
                z_s = reparameterize(mu_s, log_var_s)
                preds_s = classifier_S(z_s)
                
                mu_t, log_var_t = encoder_T(recon_t)
                z_t = reparameterize(mu_t, log_var_t)
                preds_t = classifier_T(z_t)

            l_ul_s = unlearning_loss_fn(preds_s)
            l_ul_t = unlearning_loss_fn(preds_t)

            # Total forget loss (Eq. 7)
            # 
            # and our uncertainty loss is already negative entropy, so we add.
            loss_forget = args.beta * l_r + args.gamma * (l_ul_s + l_ul_t)

            # --- Learn More Step ---
            # GLA Loss (Eq. 8)
            mu_s_recon, log_var_s_recon = encoder_S(recon_s)
            mu_t_recon, log_var_t_recon = encoder_T(recon_t)
            l_kl = kl_divergence_loss(mu_s_recon, log_var_s_recon) + \
                   kl_divergence_loss(mu_t_recon, log_var_t_recon)

            # Supervised Loss (Eq. 9)
            z_s_agnostic = reparameterize(mu_s_recon, log_var_s_recon)
            
            # For supervised loss on target, we only use the labeled part
            recon_t_labeled = recon_net(target_labeled_imgs)
            mu_t_labeled_recon, log_var_t_labeled_recon = encoder_T(recon_t_labeled)
            z_t_labeled_agnostic = reparameterize(mu_t_labeled_recon, log_var_t_labeled_recon)

            preds_agnostic_s = domain_agnostic_classifier(z_s_agnostic)
            preds_agnostic_t = domain_agnostic_classifier(z_t_labeled_agnostic)

            l_sup = classification_loss_fn(preds_agnostic_s, source_labels) + \
                    classification_loss_fn(preds_agnostic_t, target_labels)
            
            # Total learn more loss (Eq. 10)
            loss_learn_more = l_sup + args.delta * l_kl
            
            # --- Total Loss ---
            total_loss = loss_forget + loss_learn_more
            total_loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch [{epoch+1}/{args.epochs_stage2}]")
            loop.set_postfix(total_loss=total_loss.item(), loss_forget=loss_forget.item(), loss_learn=loss_learn_more.item())

    print("--- Stage 2 Finished ---")
    return recon_net, domain_agnostic_classifier


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Forget More to Learn More Implementation')
    parser.add_argument('--data_path', type=str, default='./data/OfficeHome', help='Path to dataset')
    parser.add_argument('--source', type=str, default='Art', help='Source domain')
    parser.add_argument('--target', type=str, default='Clipart', help='Target domain')
    parser.add_argument('--num_classes', type=int, default=65, help='Number of classes')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='Weight decay')
    parser.add_argument('--epochs_stage1', type=int, default=10, help='Epochs for Stage 1')
    parser.add_argument('--epochs_stage2', type=int, default=20, help='Epochs for Stage 2')
    parser.add_argument('--num_labeled', type=int, default=3, help='Number of labeled samples per class in target')
    
    # Hyperparameters from the paper
    parser.add_argument('--alpha', type=float, default=1.0, help='Weight for uncertainty loss in Stage 1')
    parser.add_argument('--beta', type=float, default=10.0, help='Weight for reconstruction loss')
    parser.add_argument('--gamma', type=float, default=0.1, help='Weight for unlearning loss')
    parser.add_argument('--delta', type=float, default=1.0, help='Weight for KL divergence loss')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Run Stage 1 ---
    e_s, c_s, e_t, c_t = train_stage1(args, device)
    
    # --- Run Stage 2 ---
    recon, classifier = train_stage2(args, device, e_s, c_s, e_t, c_t)
    
    print("\n--- Training Complete ---")
    print("Models ready for evaluation.")

    # For a target image x_t, prediction = classifier(reparameterize(*e_t(recon(x_t))))
