/*
 File: scheduler.C

 Author:
 Date  :

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
            if (clean_prev != NULL)
            {
                clean_prev->next = clean_curr->next;
                delete clean_curr;
            }
            else
            {
                delete clean_curr;
                clean_head = NULL;
            }
        }
        clean_prev = clean_curr;
        clean_curr = clean_curr->next;
    }

    Queue *n;
    Thread *thread_new;
    if (tail != NULL)
    {
        n = tail;
        tail->prev->next = NULL;
        tail = tail->prev;
    }
    else
    {
        assert(false);
    }
    thread_new = n->thread;
    delete n;

    if (Machine::interrupts_enabled())
    {
        Machine::disable_interrupts();
    }
    Thread::dispatch_to(thread_new);
    Machine::enable_interrupts();
}

void Scheduler::resume(Thread *_thread)
{
    add(_thread);
}

void Scheduler::add(Thread *_thread)
{
    assert(_thread);

    if (Thread::sched == NULL)
        Thread::register_scheduler(this);

    Queue *n = new Queue;
    n->thread = _thread;
    n->next = NULL;
    n->prev = NULL;
    if (head != NULL && tail != NULL)
    {
        n->next = head;
        head->prev = n;
        head = n;
    }
    else if (head == NULL && tail == NULL)
    {
        head = n;
        tail = n;
    }
    else
        assert(false);
}

void Scheduler::terminate(Thread *_thread)
{
    Clean *n = new Clean;
    n->thread = _thread;
    n->next = NULL;
    if (clean_head != NULL)
    {
        n->next = clean_head;
        clean_head = n;
    }
    else
    {
        clean_head = n;
    }
    yield();
}
