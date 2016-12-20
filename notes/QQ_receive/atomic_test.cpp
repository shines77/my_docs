
#include <random>
#include <unordered_map>
#include <atomic>
#include <mutex>
#include <cassert>

struct row_mutex
{
private:
    struct map_item
    {
        map_item(uint32_t _id) : id(_id), count{1}
        {
        }
        uint32_t id;
        std::atomic<uint32_t> count;
        std::mutex lock;
    };
    std::unordered_map<uint32_t, map_item *> row_lock;
    std::vector<map_item *> lock_pool;
    std::mutex g_lock;

public:
    ~row_mutex()
    {
        for(auto pair : row_lock)
        {
            delete pair.second;
        }
        for(auto ptr : lock_pool)
        {
            delete ptr;
        }
    }

    class scoped_lock
    {
    private:
        row_mutex *parent;
        map_item *item;

    public:
        scoped_lock(row_mutex &mutex, uint32_t id)
        {
            parent = &mutex;
            {
                std::unique_lock<std::mutex> l(parent->g_lock);
                auto ib = parent->row_lock.emplace(id, nullptr);
                if(ib.second)
                {
                    if(parent->lock_pool.empty())
                    {
                        ib.first->second = item = new map_item(id);
                        assert(item->count == 1);
                        assert(item->id == id);
                    }
                    else
                    {
                        ib.first->second = item = parent->lock_pool.back();
                        parent->lock_pool.pop_back();
                        assert(item->count == 0);
                        item->id = id;
                        ++item->count;
                    }
                }
                else
                {
                    item = ib.first->second;
                    assert(item->id == id);
                    ++item->count;
                }
            }
            item->lock.lock();
        }
        ~scoped_lock()
        {
            item->lock.unlock();
#if 1
            if(--item->count == 0)
            {
                std::unique_lock<std::mutex> l(parent->g_lock);
                if(item->count == 0)
                {
                    size_t c = parent->row_lock.erase(item->id);
                    assert(c == 1);
                    parent->lock_pool.emplace_back(item);
                }
            }
#else
            std::unique_lock<std::mutex> l(parent->g_lock);
            if(--item->count == 0)
            {
                size_t c = parent->row_lock.erase(item->id);
                assert(c == 1);
                parent->lock_pool.emplace_back(item);
            }
#endif
        }
    };
};

int main()
{
    row_mutex mutex;
    auto test = [&](uint32_t id)
    {
        std::mt19937 mt(id);
        std::uniform_int_distribution<uint32_t> u(0, 3);
        for(; ; )
        {
            row_mutex::scoped_lock l(mutex, u(mt));
        }
    };
    for(uint32_t i = 1; i < 4; ++i)
    {
        std::thread(test, i).detach();
    }
    test(0);
}
