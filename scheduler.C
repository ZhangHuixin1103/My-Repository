/*
    File: scheduler.C

    Author: Huixin Zhang
    Date  : 2022/11/06

 */

/*--------------------------------------------------------------------------*/
/* DEFINES */
/*--------------------------------------------------------------------------*/

/* -- (none) -- */

/*--------------------------------------------------------------------------*/
/* INCLUDES */
/*--------------------------------------------------------------------------*/

#include "scheduler.H"
#include "thread.H"
#include "console.H"
#include "utils.H"
#include "assert.H"
#include "simple_keyboard.H"
#include "blocking_disk.H"

/*--------------------------------------------------------------------------*/
/* DATA STRUCTURES */
/*--------------------------------------------------------------------------*/

/* -- (none) -- */

/*--------------------------------------------------------------------------*/
/* CONSTANTS */
/*--------------------------------------------------------------------------*/

/* -- (none) -- */

/*--------------------------------------------------------------------------*/
/* FORWARDS */
/*--------------------------------------------------------------------------*/

extern Thread *current_thread;

/*--------------------------------------------------------------------------*/
/* METHODS FOR CLASS   S c h e d u l e r  */
/*--------------------------------------------------------------------------*/

Scheduler::Scheduler()
{
    head = NULL;
    tail = NULL;
    clean_head = NULL;
    disk_head = NULL;

    Console::puts("Constructed Scheduler.\n");
}

void Scheduler::yield()
{
    Clean *clean_curr = clean_head;
    Clean *clean_prev = NULL;
    while (clean_curr != NULL)
    {
        if (clean_curr->thread != current_thread)
        {
            delete clean_curr->thread;
            if (clean_prev == NULL)
            {
                delete clean_curr;
                clean_head = NULL;
            }
            else
            {
                clean_prev->next = clean_curr->next;
                delete clean_curr;
            }
        }
        clean_prev = clean_curr;
        clean_curr = clean_curr->next;
    }

    Queue *node;
    Thread *thread_new;
    if (tail != NULL)
    {
        node = tail;
        tail->prev->next = NULL;
        tail = tail->prev;
    }
    else
    {
        assert(false);
    }
    thread_new = node->thread;
    delete node;

    if (Machine::interrupts_enabled())
    {
        Machine::disable_interrupts();
    }
    Thread::dispatch_to(thread_new);
    Machine::enable_interrupts();
}

void Scheduler::resume(Thread *_thread)
{
    Disk *disk_inst = disk_head;
    BlockingDisk *disk;
    while (disk_inst != NULL)
    {
        disk = disk_inst->disk;
        Thread *thread = disk->request_head();
        if (thread != NULL)
        {
            bool flag = disk->is_ready() || disk->disk_status;
            if (flag)
            {
                add(thread);
            }
        }
        disk_inst = disk_inst->next;
    }
    add(_thread);
}

void Scheduler::add(Thread *_thread)
{
    assert(_thread);

    if (Thread::sched == NULL)
        Thread::sched_register(this);

    Queue *node = new Queue;
    node->thread = _thread;
    node->next = NULL;
    node->prev = NULL;
    if (head != NULL && tail != NULL)
    {
        node->next = head;
        head->prev = node;
        head = node;
    }
    else if (head == NULL && tail == NULL)
    {
        head = node;
        tail = node;
    }
    else
        assert(false);
}

void Scheduler::terminate(Thread *_thread)
{
    Clean *node = new Clean;
    node->thread = _thread;
    node->next = NULL;
    if (clean_head != NULL)
    {
        node->next = clean_head;
        clean_head = node;
    }
    else
    {
        clean_head = node;
    }
    yield();
}

void Scheduler::disk_register(BlockingDisk *_disk)
{
    Disk *disk_inst = new Disk;
    disk_inst->disk = _disk;
    disk_inst->next = NULL;
    if (disk_head != NULL)
    {
        disk_inst->next = disk_head;
        disk_head = disk_inst;
    }
    else
    {
        disk_head = disk_inst;
    }
}
