export PATH=$PATH:.


# git current branch on prompt
GIT_PS1_SHOWUPSTREAM="yes"
GIT_PS1_SHOWCOLORHINTS="yes"
GIT_PS1_SHOWDIRTYSTATE="yes"
source ~/.git-prompt.sh
precmd () { __git_ps1 "[" "]%~$ " "%s" }
# git completions
zstyle ':completion:*:*:git:*' script ~/repos/gh/git/git/contrib/completion/git-completion.zsh


# history
HISTSIZE=700
SAVEHIST=700
HISTFILE=~/.history
setopt APPEND_HISTORY
setopt INC_APPEND_HISTORY
#setopt SHARE_HISTORY

# HADOOP
export HADOOP_HOME=/usr/local/Cellar/hadoop/2.7.0
export HADOOP_CONF_DIR=$HADOOP_HOME/libexec/etc/hadoop

alias hdstart='/usr/local/Cellar/hadoop/2.7.0/sbin/start-dfs.sh'
alias hdstop='/usr/local/Cellar/hadoop/2.7.0/sbin/stop-dfs.sh'

alias hystart='/usr/local/Cellar/hadoop/2.7.0/sbin/start-yarn.sh'
alias hystop='/usr/local/Cellar/hadoop/2.7.0/sbin/stop-yarn.sh'

# HIVE
export HIVE_HOME=/usr/local/Cellar/hive/apache-hive-1.2.0-bin
export PATH=$PATH:$HIVE_HOME/bin

#GHC
export PATH=$PATH:$HOME/Library/Haskell/bin

# AWS
source /usr/local/share/zsh/site-functions/_aws
alias ssh_aws='ssh `cat ~/repos/gh/bekbolatov/kaggle/scripts/master`'
alias sbt_aws='sbt assembly && scp target/scala-2.10/AvitoProject-assembly-1.0.jar `cat ../../scripts/master`:/home/hadoop/.'

