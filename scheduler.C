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

/* -- (none) -- */

/*--------------------------------------------------------------------------*/
/* METHODS FOR CLASS   S c h e d u l e r  */
/*--------------------------------------------------------------------------*/

Scheduler::Scheduler()
{
    que_size = 0;
    Console::puts("Constructed Scheduler.\n");
}

void Scheduler::yield()
{
    que_size--;
    Thread *thread_new = que_ready.dequeue();
    Thread::dispatch_to(thread_new);
}

void Scheduler::resume(Thread *_thread)
{
    que_ready.enqueue(_thread);
    que_size++;
}

void Scheduler::add(Thread *_thread)
{
    que_ready.enqueue(_thread);
    que_size++;
}

void Scheduler::terminate(Thread *_thread)
{
    for (int i = 0; i < que_size; i++)
    {
        Thread *thread_top = que_ready.dequeue();

        if (_thread->ThreadId() == thread_top->ThreadId())
        {
            que_size--;
        }
        else
        {
            que_ready.enqueue(thread_top);
        }
    }
}
