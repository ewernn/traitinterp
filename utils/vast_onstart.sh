#!/bin/bash
# Vast.ai on-start provisioning script

# Disable tmux
touch /root/.no_auto_tmux

# Install neovim (latest stable)
apt-get update && apt-get install -y software-properties-common
add-apt-repository -y ppa:neovim-ppa/stable
apt-get update && apt-get install -y neovim

# Create dev user
useradd -m -s /bin/bash dev
echo "dev ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
touch /home/dev/.no_auto_tmux
chown dev:dev /home/dev/.no_auto_tmux

# Persist env vars for all users (from account-level vars)
cat >> /etc/environment << EOF
GITHUB_TOKEN=${GITHUB_TOKEN}
R2_ACCESS_KEY_ID=${R2_ACCESS_KEY_ID}
R2_SECRET_ACCESS_KEY=${R2_SECRET_ACCESS_KEY}
R2_ENDPOINT=${R2_ENDPOINT}
HF_TOKEN=${HF_TOKEN}
OPENAI_API_KEY=${OPENAI_API_KEY}
HF_HOME=/home/dev/.cache/huggingface
EOF

# Install Claude Code
su - dev -c 'curl -fsSL https://claude.ai/install.sh | bash'

# Git config
su - dev -c 'git config --global user.name "ewernn"'
su - dev -c 'git config --global user.email "ewernn@users.noreply.github.com"'

# Clone repos
su - dev -c "git clone https://${GITHUB_TOKEN}@github.com/ewernn/trait-interp.git ~/trait-interp"
su - dev -c "git clone https://${GITHUB_TOKEN}@github.com/ewernn/cc-plugins.git ~/cc-plugins"
su - dev -c "mkdir -p ~/.claude/plugins && ln -s ~/cc-plugins/r ~/.claude/plugins/r"

# Bashrc
cat >> /home/dev/.bashrc << 'EOF'
export PATH="$HOME/.local/bin:$PATH"
source ~/trait-interp/.venv/bin/activate 2>/dev/null || true
alias cla='claude --dangerously-skip-permissions'
alias clar='claude --dangerously-skip-permissions --resume'
alias cl='clear'
EOF
chown dev:dev /home/dev/.bashrc

# Neovim config (file browser + basics)
mkdir -p /home/dev/.config/nvim
cat > /home/dev/.config/nvim/init.lua << 'NVIMEOF'
vim.opt.number = true
vim.opt.relativenumber = false
vim.opt.mouse = "a"
vim.opt.clipboard = "unnamedplus"
vim.opt.tabstop = 2
vim.opt.shiftwidth = 2
vim.opt.expandtab = true
vim.opt.scrolloff = 8
vim.opt.termguicolors = true
vim.g.mapleader = " "
vim.keymap.set("n", "<C-s>", ":w<CR>", { desc = "Save" })
vim.keymap.set("i", "<C-s>", "<Esc>:w<CR>a", { desc = "Save" })
vim.keymap.set("i", "jk", "<Esc>", { desc = "Exit insert mode" })
vim.keymap.set("n", "<leader>e", ":Neotree toggle<CR>", { desc = "Toggle file tree" })
local lazypath = vim.fn.stdpath("data") .. "/lazy/lazy.nvim"
if not vim.loop.fs_stat(lazypath) then
  vim.fn.system({"git", "clone", "--filter=blob:none", "https://github.com/folke/lazy.nvim.git", "--branch=stable", lazypath})
end
vim.opt.rtp:prepend(lazypath)
require("lazy").setup({
  {"folke/tokyonight.nvim", priority = 1000, config = function() vim.cmd.colorscheme("tokyonight") end},
  {"nvim-neo-tree/neo-tree.nvim", branch = "v3.x", dependencies = {"nvim-lua/plenary.nvim", "nvim-tree/nvim-web-devicons", "MunifTanjim/nui.nvim"}, config = function() require("neo-tree").setup({close_if_last_window = true, window = {position = "right"}, filesystem = {filtered_items = {visible = true, hide_dotfiles = false, hide_gitignored = false}}}) end},
  {"nvim-lualine/lualine.nvim", config = function() require("lualine").setup() end},
})
NVIMEOF
chown -R dev:dev /home/dev/.config

# Install rclone and configure R2
curl https://rclone.org/install.sh | bash
mkdir -p /home/dev/.config/rclone
cat > /home/dev/.config/rclone/rclone.conf << EOF
[r2]
type = s3
provider = Cloudflare
access_key_id = ${R2_ACCESS_KEY_ID}
secret_access_key = ${R2_SECRET_ACCESS_KEY}
endpoint = ${R2_ENDPOINT}
acl = private
EOF
chown -R dev:dev /home/dev/.config/rclone

# Setup venv and install deps
su - dev -c "cd ~/trait-interp && pip3 install --break-system-packages uv && uv venv && uv pip install -r requirements.txt"

# Pull data from R2
su - dev -c "cd ~/trait-interp && ./utils/r2_pull.sh"
