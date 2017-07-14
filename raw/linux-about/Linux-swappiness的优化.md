
/proc/sys/vm/swappiness 这个值表示内存还剩多少百分比的时候开始使用 swap 交互分区，默认值是 60：

	$ cat /proc/sys/vm/swappiness
	60

可以修改为 10，这样系统在系统内存还剩 10％ 的时候才会使用 swap 交互分区，以提高内存的效率：

	$ sudo sysctl vm.swappiness=10
