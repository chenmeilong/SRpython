# 志不坚者智不达
#
# ssh 密钥
#
# RSA -非对称密钥验证
#
# 公钥 public  key
#
#
# 私钥  private key
#
#
# 10.0.0.31    -----> 10.0.0.41
# 私钥                    公钥
#
#
#
#
# 421
# rwx     rwx     rwx
# 属主     属组   others
#
#
# ssh-rsa AAAAB3NzaC1yc2EAAAABIwAAAQEAvNhNa1INz7Dqhq5BOu8yqvvjVCguO4iM1bH6SZHFu418pKMJwb4AU2qeuDI3lkZjPOmjQ7dap3m9f8W/jCb4tU4rfTj39m98BJMgTotHMHhENqh7+nPa30cDuLDXM84XylnF3u1/ZW7NdupCzbG4Y1Tsof6QuNYd2fn07+ZpOBMUJIssJHXb1HCPTzp5eA8j9wKPIPHk/ASjjUJk5TIc6r9SJmgNQN1J3xRB7IWKoesYiEJk6K/VoDHDY355ozsNKm0QTYT39q4KUmIc8c8g5Dl51u05jYZqlUYN1+1pqdzl81xhaqj+3aRRZNn36RvCG9NfwH1yWwzHSsqcY37Mrw== root@nfs01
#
#
#
#
#
# 进程: qq 要以一个整体的形式暴露给操作系统管理，里面包含对各种资源的调用，内存的管理，网络接口的调用等。。。
# 对各种资源管理的集合 就可以成为  进程
#
# 线程: 是操作系统最小的调度单位， 是一串指令的集合
#
#
# 进程 要操作cpu , 必须要先创建一个线程  ，
# all the threads in a process have the same view of the memory
# 所有在同一个进程里的线程是共享同一块内存空间的
#
#
# A thread线程 is an execution执行 context上下文, which is all the information a CPU needs to execute a stream of instructions.
# 线程就是cpu执行时所需要的
# Suppose you're reading a book, and you want to take a break right now, but you want to be able to come back and resume reading from the exact point where you stopped. One way to achieve that is by jotting down the page number, line number, and word number. So your execution context for reading a book is these 3 numbers.
# 假设
# If you have a roommate, and she's using the same technique, she can take the book while you're not using it, and resume reading from where she stopped. Then you can take it back, and resume it from where you were.
#
# Threads work in the same way. A CPU is giving you the 幻觉illusion that it's doing multiple多 computations运算 at the same time. It does that by spending花费 a bit点 of time on each computation运算. It can do that because it has an execution context for each computation. Just like you can share a book with your friend, many tasks can share a CPU.
#
# On a more technical level, an execution context (therefore a thread) consists组合 of the values of the CPU's registers 寄存器.
#
# Last: threads are different from 进程processes. A thread is a context of execution, while a process is a bunch一簇，一堆 of resources资源 associated相关的 with a computation. A process can have one or many threads.
#
# Clarification: the resources associated with a process include memory pages (all the threads in a process have the same view of the memory), file descriptors (e.g., open sockets), and security credentials (e.g., the ID of the user who started the process).
#
#
#
# 什么是进程(process)？
# An executing instance of a program is called a process.
#
# Each process provides the resources needed to execute a program.
# A process has a virtual虚拟 address space, executable code,
#  open handles to system objects, a security context,
#  a unique唯一的 process进程标识符，pid identifier, environment variables,
#  a priority 优先级类class, minimum and maximum working set sizes,
#   and at least至少 one thread线程 of execution.
#   Each process is started with a single thread,
#    often called the primary主 thread, but can create additional额外的
#     threads from any of its threads.
#
#
# 进程与线程的区别？
# Threads share the address space of the process that created it; processes have their own address space.
# 线程共享内存空间，进程的内存是独立的
# Threads have direct access to the data segment of its process; processes have their own copy of the data segment of the parent process.
#
# Threads can directly communicate with other threads of its process; processes must use interprocess communication to communicate with sibling processes.
# 同一个进程的线程之间可以直接交流，两个进程想通信，必须通过一个中间代理来实现
#
# New threads are easily created; new processes require duplication of the parent process.
# 创建新线程很简单， 创建新进程需要对其父进程进行一次克隆
#
# Threads can exercise considerable control over threads of the same process; processes can only exercise control over child processes.
# 一个线程可以控制和操作同一进程里的其他线程，但是进程只能操作子进程
#
# Changes to the main thread (cancellation, priority change, etc.) may affect the behavior of the other threads of the process; changes to the parent process does not affect child processes.
#
# locks = {
#     door1:key1,
#     door2:key2,
# }
#
#
# file1
# a
# b
# c
# d
# e
# f
#
#
#
#
# redLight = False
#
# while True：
#     if counter >30:
#        redLight = True
#     if counter >50 :
#        redLight = False
#        counter = 0
#
#
# event.wait()
#
# # a server thread can set or reset it
# event.set()
# event.clear()
# If the flag is set, the wait method doesn’t do anything.
# 标志位设定了，代表绿灯，直接通行
# If the flag is cleared, wait will block until it becomes set again.
# 标志位被清空，代表红灯，wait等待变绿灯
# Any number of threads may wait for the same event.
